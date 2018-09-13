import numpy as np
import torch
import torchvision.transforms as transforms

import beta_set as bs
import evaluation_lib_par as el
import stan_lib_x as slx

import pandas as pd

# Get configuration
conf = el.get_conf()
num_proc = int(conf['num_proc'])
h5_file = conf['h5_file']
max_frame_cache = int(conf['max_frame_cache'])
cuda = bool(int(conf['cuda']))

# Parameters
all_ascales = [0.225]
all_beams = [50.]
all_pos_scales = [True]

# Dataset
dataset_mode = 'isolated'
all_subsets = ['et05_caf_real', 'et05_ped_real', 'et05_str_real', 'et05_bus_real']

# Define Network
net_file = '/media/brauns/data/Dropbox/repos/stn_private/pytorch/results/models/stan/best'
net = slx.audio_stan(num_sensors=6, inp_size=[123] * 6, tra_size=50,
                     att_size=20, cla_size=350, cla_layers=5,
                     num_classes=59,
                     tra_type='identity', rnn_mode='LSTM', cla_dropout=0.3)

# Load network
net.load_state_dict(torch.load(net_file, map_location=lambda storage, loc: storage))
net.eval()
if cuda == True:
    net.cuda()

# all_disabled = reversed([range(0, i) for i in range(0, 6)])
all_disabled = [range(0,i) for i in range(0,6)]
# all_disabled.extend([[x] for x in range(6)])
# all_disabled=[[0,1,3]]
df = pd.DataFrame(columns=['Type', 'Subset', 'Disabled', 'WER', 'CER'])


# Loop over subsets
for subset in all_subsets:

    for disabled in all_disabled:
        print(disabled)
        net.merge_module.disabled = disabled

        # Create dataset, sampler, loader
        composed = transforms.Compose([bs.warp_ctc_shift(), bs.standardization('sample')])
        testset = bs.dset(h5file=h5_file, dataset_mode=dataset_mode, subset=subset, transform=composed)
        sampler = bs.HighThroughputSampler(testset, shuffle_batches=False, num_splits=1, max_frames=max_frame_cache,
                                           debug=0, roll=False)
        batch_sampler = el.SimpleBatchSampler(sampler=sampler)
        test_loader = torch.utils.data.DataLoader(testset, batch_sampler=batch_sampler, num_workers=0,
                                                  collate_fn=bs.collate_fn)

        filename = '{}_dis{}'.format(subset, '-'.join(map(str,disabled)))
        print(filename)

        # Get network output
        inference_list = el.infer(model=net, dataloader=test_loader, cuda=cuda)
        PER, WER, CER = el.error_preliminary(inference_list, filename)
        print(PER,WER,CER)

        pd_dict = {}
        pd_dict['Type'] = net_file.split('/')[-2]
        pd_dict['WER'] = WER
        pd_dict['CER'] = CER
        pd_dict['Disabled'] = disabled
        pd_dict['Subset'] = subset
        row_df=pd.DataFrame.from_dict([pd_dict])
        df=df.append(row_df, ignore_index=True)

        for pos_scale in all_pos_scales:

            kw = el.kaldi_converter(inference_list, filename=filename, scale=pos_scale)
            logprob = kw.write_logprob()
            reference = kw.write_reference()

            for beam in all_beams:
                for ascale in all_ascales:
                    dec = el.decoder(logprob=logprob, num_proc=num_proc, acoustic_scale=ascale, beam=beam)
                    dec.full()
df = df[['Type', 'Subset','Disabled', 'WER', 'CER']]
df.to_csv('decoder/temp/df.csv', index=None)
