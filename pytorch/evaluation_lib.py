import csv
import os
import subprocess
from collections import OrderedDict
from timeit import default_timer as timer

import numpy as np
import tables
import torch
from seqtools import kaldi_io, metric
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.transforms as transforms
from datetime import datetime


import stan_lib_x as slx
from beta_set import HighThroughputSampler, SimpleBatchSampler, dset, collate_fn
from transform_lib import standardization, warp_ctc_shift


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def infer(model, dataloader, cuda=True):
    inference_list = []
    for sensor_list, feature_length, label, label_length, debug in tqdm(dataloader):


        # Convert to variables
        if cuda == True:
            sensor_list = [Variable(sensor).cuda() for sensor in sensor_list]
        else:
            sensor_list = [Variable(sensor) for sensor in sensor_list]

        # Network output
        output, _ = model(sensor_list, 0)  # output is seqlen x batch x features
        output = output.cpu().data.numpy().swapaxes(0, 1)  # swap to batch x seqlen x features

        # Log probabilities
        output_sm = softmax(output, axis=2)
        logprob = np.log(output_sm)

        # Write sample-wise to inference-list
        for sample, sample_length, sample_label, debug_dict in zip(logprob, feature_length, label, debug['debug']):
            sample_dict = {}
            sample_dict['logprob'] = sample[:sample_length, :]
            sample_dict['key'] = debug_dict['key']
            sample_dict['label_string'] = debug_dict['label_string']
            sample_dict['label'] = debug_dict['label']
            sample_dict['label_length'] = debug_dict['label_length']
            inference_list.append(sample_dict)

            # print(sample_dict['label_string'])
    return inference_list


def error_preliminary(inference_list, subset='Not defined'):
    test_meter = metric.meter(blank=0)

    for sample_dict in inference_list:
        output_sm = np.expand_dims(np.exp(sample_dict['logprob']), axis=0).swapaxes(0,
                                                                                    1)  # convert to time x batch x features
        test_meter.extend_guessed_labels(output_sm)
        test_meter.extend_target_labels(bY=sample_dict['label'], b_lenY=sample_dict['label_length'])

    for a, b in zip(test_meter.target_labels, test_meter.guessed_labels):
        print "{} // {}".format(a, b)

    PER, WER, CER = test_meter.get_metrics()

    row = OrderedDict()
    row['Subset'] = subset
    row['WER'] = round(WER*100,2)
    row['SER'] = round(PER*100,2)
    row['CER'] = round(CER*100,2)
    row['Time'] = str(datetime.now())

    csv_file = os.path.join('decoder', 'temp', 'results_prel.csv')
    with open(csv_file, 'a') as f:
        writer = csv.DictWriter(f, row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    return PER, WER, CER


class kaldi_converter():
    def __init__(self, inference_list, filename='default', scale=False):
        # Set parameters
        self.inference_list = inference_list
        self.filename = filename
        self.scale = scale

        # Get parameters
        self.root = os.path.dirname(os.path.realpath(__file__))
        self.temp = os.path.join(self.root, 'decoder', 'temp')

        self.logprob = os.path.join(self.temp, self.filename + '.prob')  # output 1
        self.reference = os.path.join(self.temp, self.filename + '.ref')  # output 2

    def write_logprob(self):

        # Write sample-wise
        with open(self.logprob, 'w') as f:
            for sample_dict in self.inference_list:
                if self.scale == False:
                    kaldi_io.write_mat(f, sample_dict['logprob'], key=sample_dict['key'][0])
                else:
                    print('Scaling with tr05_simu_real prior')
                    prior = np.load('decoder/files/tr05_simu_real.npy')
                    log_prior = np.log(prior)
                    log_prior[log_prior == -np.inf] = 0

                    scaled = sample_dict['logprob'] - log_prior[None, :]
                    kaldi_io.write_mat(f, scaled, key=sample_dict['key'][0])


                    # fig, (ax1,ax2) = plt.subplots(2)
                    # vmax=max(np.max(scaled), np.max(sample_dict['logprob']))
                    # vmin=min(np.min(scaled), np.min(sample_dict['logprob']))
                    #
                    # im1=ax1.imshow(sample_dict['logprob'].T, aspect='auto', vmin=vmin, vmax=vmax)
                    # plt.colorbar(im1, ax=ax1)
                    # im2=ax2.imshow(scaled.T, aspect='auto', vmin=vmin, vmax=vmax)
                    # plt.colorbar(im2, ax=ax2)
                    # plt.show()

        return self.logprob

    def write_reference(self):
        with open(self.reference, 'w') as f:
            for sample_dict in self.inference_list:
                row = sample_dict['key'][0] + ' ' + sample_dict['label_string'][0] + '\n'
                f.write(row)
        return self.reference

    def full(self):
        self.write_logprob()
        self.write_reference()


class decoder():
    def __init__(self, logprob, acoustic_scale=0.9, beam=50,
                 num_proc=2):
        # Get configuration
        conf = get_conf()

        # Set parameters
        self.decoderbin = conf['decoderbin']
        self.acoustic_scale = acoustic_scale
        self.beam = beam
        self.num_proc = num_proc

        # Get parameters
        self.root = os.path.dirname(os.path.realpath(__file__))  # script root
        self.files = os.path.join(self.root, 'decoder', 'files')
        self.temp = os.path.join(self.root, 'decoder', 'temp')

        self.tlg = os.path.join(self.files, 'TLG.fst')  # .fst - language model
        self.word_table = os.path.join(self.files, 'words.txt')  # .txt - word table
        self.int2sym_script = os.path.join(self.files, 'int2sym.pl')  # .pl script - convert ints to symbols (=words)

        # Inputs
        self.logprob = logprob
        self.reference = os.path.splitext(self.logprob)[0] + '.ref'

        # Time
        self.tstart = timer()

    def decode_faster_parallel(self):
        # Combined output file
        self.transcriptions_int = os.path.splitext(self.logprob)[0] + '.int'

        # Read and split logprob
        if self.num_proc > 1:
            logprob = [(key, mat) for key, mat in kaldi_io.read_mat_ark(self.logprob)]
            logprob_split = chunkIt(logprob, self.num_proc)

            num_split = 0
            decode_queue = []
            for split in logprob_split:
                filename = os.path.join(self.temp, 'part{}.prob'.format(num_split))
                with open(filename, 'w') as f:
                    for key, mat in split:
                        kaldi_io.write_mat(f, mat, key=key)
                num_split += 1
                decode_queue.append(filename)
        else:
            decode_queue = [self.logprob]

        # Loop over decoding processes
        all_processes = []
        all_output_files = []
        for input_file in decode_queue:
            # Define output file
            output_file = os.path.splitext(input_file)[0] + '.int'
            all_output_files.append(output_file)

            # Decode
            all_processes.append(subprocess.Popen(
                ['{}/decode-faster'.format(self.decoderbin),
                 '--beam-delta=5.',
                 '--max-active=7000',
                 '--min-active=20',
                 '--word-symbol-table={}'.format(self.word_table),
                 '--acoustic-scale={}'.format(self.acoustic_scale),
                 '--beam={}'.format(self.beam),
                 self.tlg,
                 'ark:{}'.format(input_file),
                 'ark,t:{}'.format(output_file)],
                shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

        # Wait for all processes to finish
        for process in all_processes:
            dec_out, dec_err = process.communicate()
            print(dec_out, dec_err)

        # Concatenate to a single output file ONLY if multiprocessing was used
        if self.num_proc > 1:
            with open(self.transcriptions_int, 'wb') as outfile:
                for f in all_output_files:
                    with open(f, 'rb') as infile:
                        outfile.write(infile.read())

        return self.transcriptions_int

    def int2sym(self):

        # Output file
        self.transcriptions_sym = os.path.splitext(self.transcriptions_int)[0] + '.sym'

        # Convert integers to symbols (=words)
        command = 'cat {} | {} -f 2- {} | sed \'s:<UNK>::g\' | sed \'s:<NOISE>::g\' | sed \'s:<SPOKEN_NOISE>::g\' > {}'.format(
            self.transcriptions_int, self.int2sym_script, self.word_table, self.transcriptions_sym)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

        return self.transcriptions_sym

    def compute_wer(self):

        wer_process = subprocess.Popen(
            ['{}/compute-wer'.format(self.decoderbin),
             '--text',
             '--mode=present',
             'ark:{}'.format(self.reference),
             'ark:{}'.format(self.transcriptions_sym)],
            shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        wer_out, wer_err = wer_process.communicate()

        try:
            self.WER = float(wer_out.split()[1])
            self.SER = float(wer_out.split()[14])
        except:
            self.WER = 'FAIL'
            self.SER = 'FAIL'
        self.output = wer_out

        return self.WER

    def log_to_csv(self):
        row = OrderedDict()
        row['filename'] = os.path.basename(self.logprob)
        row['WER'] = self.WER
        row['SER'] = self.SER
        row['acoustic_scale'] = self.acoustic_scale
        row['beam'] = self.beam
        row['num_proc'] = self.num_proc
        row['runtime'] = round(timer() - self.tstart, 2)
        row['time'] = str(datetime.now())
        row['output'] = ''.join(self.output.splitlines())

        csv_file = os.path.join(self.temp, 'results.csv')
        with open(csv_file, 'a') as f:
            writer = csv.DictWriter(f, row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def full(self):
        self.decode_faster_parallel()
        self.int2sym()
        self.compute_wer()
        self.log_to_csv()


def get_prior(dataset='tr05_simu_real'):
    conf = get_conf()
    h5_file = conf['h5_file']

    with tables.open_file(h5_file, 'r') as f:
        label = f.get_node(os.path.join(os.sep, 'isolated', dataset, 'ch1', 'label'))
        all_labels = np.concatenate(list(label))

        # Get counts
        occ_label = np.bincount(all_labels)
        occ_blank = all_labels.size + len(label)

        # Stack
        occ = np.hstack((np.asarray([occ_blank]), occ_label))

        # Convert to probability
        prob = occ / float(np.sum(occ))

        np.save('decoder/files/{}'.format(dataset), prob)


def get_conf():
    d = {}
    with open('decoder/files/conf', 'r') as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            (key, val) = line.split('=')
            d[key] = val
    return d


if __name__ == '__main__':

    # Get configuration
    conf = get_conf()
    num_proc = int(conf['num_proc'])
    h5_file = conf['h5_file']
    max_frame_cache = int(conf['max_frame_cache'])
    cuda = bool(int(conf['cuda']))

    # Parameters
    all_ascales = np.linspace(0.3, 0.1, 5)
    all_beams = [50., 100.]
    all_pos_scales = [True, False]

    # Dataset
    dataset_mode = 'isolated'
    all_subsets = ['et05_caf_real', 'et05_ped_real', 'et05_str_real', 'et05_bus_real']

    # Define Network
    net_file = '/media/stefbraun/data/Dropbox/repos/stn_private/pytorch/models/LSTM_big/6x_fbank_seed2_cla_layers5_cla_size350_cla_dropout0.3_weight_decay_0.0001_standardization_sample/best'
    net = slx.audio_stan(num_sensors=6, inp_size=[123] * 6, tra_size=50,
                         att_size=20, cla_size=350, cla_layers=5,
                         num_classes=59,
                         tra_type='identity', rnn_mode='LSTM', cla_dropout=0.3)

    # Load network
    net.load_state_dict(torch.load(net_file, map_location=lambda storage, loc: storage))
    net.eval()
    if cuda == True:
        net.cuda()

    # Loop over subsets
    for subset in all_subsets:

        # Create dataset, sampler, loader
        composed = transforms.Compose([warp_ctc_shift(), standardization('sample')])
        testset = dset(h5file=h5_file, dataset_mode=dataset_mode, subset=subset, transform=composed)
        sampler = HighThroughputSampler(testset, shuffle_batches=False, num_splits=1, max_frames=max_frame_cache,
                                        debug=0, roll=False)
        batch_sampler = SimpleBatchSampler(sampler=sampler)
        test_loader = torch.utils.data.DataLoader(testset, batch_sampler=batch_sampler, num_workers=0,
                                                  collate_fn=collate_fn)

        # Get network output
        inference_list = infer(model=net, dataloader=test_loader, cuda=cuda)
        PER, WER, CER = error_preliminary(inference_list, subset)
        for pos_scale in all_pos_scales:

            kw = kaldi_converter(inference_list, filename='{}_{}'.format(subset, pos_scale), scale=pos_scale)
            logprob = kw.write_logprob()
            reference = kw.write_reference()

            for beam in all_beams:
                for ascale in all_ascales:
                    dec = decoder(logprob=logprob, num_proc=num_proc, acoustic_scale=ascale, beam=beam)
                    dec.full()
