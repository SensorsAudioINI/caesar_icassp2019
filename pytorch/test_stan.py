import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/include:/usr/local/cuda-8.0/lib64/:' + \
                                os.environ['LD_LIBRARY_PATH']

import numpy as np
import torch
import torchvision.transforms as transforms

import beta_set as bs
import evaluation_lib as el
import stan_lib_x as slx
import transform_lib as tl

# Get configuration
# conf = el.get_conf()
# num_proc = int(conf['num_proc'])
# h5_file = conf['h5_file']
h5_file = '/Data/Dropbox/tidigits_kaldi_fbank.h5'
# max_frame_cache = int(conf['max_frame_cache'])
# cuda = bool(int(conf['cuda']))
cuda = True
# Parameters
all_ascales = np.linspace(0.3, 0.1, 5)
all_beams = [50., 100.]
all_pos_scales = [True, False]

# Dataset
dataset_mode = 'default'
all_subsets = ['test']

# Define Network
net_file = '/Data/Dropbox/PhD/Projects/caesar_iscas2019/pytorch/models/debug/1x_debug_bn/best'
net = slx.audio_stan(num_sensors=1, inp_size=[123], tra_size=50,
                     att_size=20, cla_size=320, cla_layers=4,
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
    composed = transforms.Compose([tl.warp_ctc_shift(), tl.standardization('sample')])
    testset = bs.dset(h5file=h5_file, dataset_mode=dataset_mode, subset=subset, transform=composed)
    sampler = bs.HighThroughputSampler(testset, shuffle_batches=False, num_splits=1, max_frames=10000,
                                       debug=0, roll=False)
    batch_sampler = el.SimpleBatchSampler(sampler=sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_sampler=batch_sampler, num_workers=0,
                                              collate_fn=bs.collate_fn)

    # Get network output
    inference_list = el.infer(model=net, dataloader=test_loader, cuda=cuda)
    PER, WER, CER = el.error_preliminary(inference_list, subset)
    print(PER,WER,CER)
    # for pos_scale in all_pos_scales:
    #
    #     kw = el.kaldi_converter(inference_list, filename='{}_{}'.format(subset, pos_scale), scale=pos_scale)
    #     logprob = kw.write_logprob()
    #     reference = kw.write_reference()
    #
    #     for beam in all_beams:
    #         for ascale in all_ascales:
    #             dec = el.decoder(logprob=logprob, num_proc=num_proc, acoustic_scale=ascale, beam=beam)
    #             dec.full()
