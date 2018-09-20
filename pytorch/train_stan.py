from __future__ import print_function

import matplotlib

matplotlib.use('Agg')  # from https://stackoverflow.com/questions/27147300/how-to-clean-images-in-python-django
from warpctc_pytorch import CTCLoss
import torch.optim as optim

import argparse
import collections
import os
from timeit import default_timer as timer

import numpy as np
from seqtools import progress, metric, log

import stan_lib_x as slx

from torch.autograd import Variable
import torch.nn.init as weight_init
import torch.utils.data.dataloader
from beta_set import dset, collate_fn, HighThroughputSampler, SimpleBatchSampler, max_frame_batch
import transform_lib as tl
import torchvision.transforms as transforms
from tqdm import tqdm

from logger import Logger
from plot_lib import plot_image, multi_plot_an

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Double Audio STAN network.')
    # File and path naming stuff
    parser.add_argument('--dataset',
                        # default='/media/stefbraun/cache/TIDIGITS_MFCC_25fps.h5',
                        # default='/media/brauns/cache/CHiME.h5',
                        # default='/Data/Dropbox/tidigits_kaldi_fbank.h5',
                        default='/Data/DATASETS/CAESAR_TIDIGITS/tidigits_spk_sep.h5',
                        # default='/home/stefbraun/Downloads/CHiME_mfcc.h5',
                        help='HDF5 File that has the dataset')
    parser.add_argument('--dataset_mode',
                        default='default')
    parser.add_argument('--trainset',
                        default='train')
    parser.add_argument('--valset',
                        default='test')

    parser.add_argument('--run_id', default=os.environ.get('LSB_JOBID', 'default'),
                        help='ID of the run, used in saving.  Gets job ID on Euler, otherwise is "default".')
    parser.add_argument('--experiment', default='train_direct_spk', help='Name of experiment for log and model folders')
    parser.add_argument('--filename', default='train_direct_spk_bn',
                        help='Filename to save model and log to.')

    # Control meta parameters
    parser.add_argument('--seed', default=11, type=int,
                        help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--max_frame_cache', default=1000, type=int, help='Max frames per batch.')
    parser.add_argument('--num_epochs', default=150, type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience', default=150, type=int,
                        help='How long to wait for an increase in validation loss before quitting.')
    parser.add_argument('--wait_period', default=1, type=int,
                        help='How long to wait before looking for early stopping.')

    # Training params
    parser.add_argument('--num_sensors', default=1, type=int, help='Number of sensors')
    parser.add_argument('--att_size', default=20, type=int, help='Size of the RNN in the attention module')
    parser.add_argument('--tra_size', default=50, type=int, help='Size of transformation layer')
    parser.add_argument('--tra_type', default='identity', help='Type of transformation')
    parser.add_argument('--classes', default=11, type=int, help='Number of target classes (including blank label)')
    parser.add_argument('--cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--cla_dropout', default=0.3, type=float, help='Dropout in classification layer')
    parser.add_argument('--rnn_type', default='LSTM', help='RNN type in all RNN-layers')
    parser.add_argument('--cla_layers', default=4, type=int, help='number of classification layers')
    parser.add_argument('--cla_size', default=320, type=int, help='Size of classification layer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight of L2 regularizer')
    parser.add_argument('--standardization', default='epoch', help='Standardization mode')
    parser.add_argument('--concatenation', default=0, type=int, help='Concatenate sensors')
    parser.add_argument('--att_share', default=False, type=bool, help='Attention module weight sharing')

    args = parser.parse_args()
    print(args)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create savers, loggers
    comb_filename = '_'.join([str(args.num_sensors) + 'x', args.filename])

    savedir = 'models/{}/{}'.format(args.experiment, comb_filename)  # save model directory
    logfile = 'models/{}/{}/logfile.csv'.format(args.experiment, comb_filename)  # .csv logfile

    logdir = 'logs/{}/{}'.format(args.experiment, comb_filename)  # tensorboard logger directory
    logger = Logger(logdir)  # tensorboard logger object
    print('Log directory = {}'.format(logdir))

    try:
        os.makedirs(savedir)
    except:
        pass
    log_dict = collections.OrderedDict()

    # Prepare transforms
    train_transforms = [tl.warp_ctc_shift(), tl.gaussian_noise()]
    val_transforms = [tl.warp_ctc_shift(), ]
    if args.concatenation == True:
        train_transforms.append(tl.concatenation())
        val_transforms.append(tl.concatenation())
    train_composed = transforms.Compose(train_transforms)
    val_composed = transforms.Compose(val_transforms)

    # Create datasets
    trainset = dset(h5file=args.dataset, dataset_mode=args.dataset_mode, subset=args.trainset, speaker_wise=False,
                    transform=train_composed)
    valset = dset(h5file=args.dataset, dataset_mode=args.dataset_mode, subset=args.valset, speaker_wise=False,
                  transform=val_composed)

    # Define sampler
    train_sampler = HighThroughputSampler(trainset, shuffle_batches=True, num_splits=3, max_frames=args.max_frame_cache,
                                          debug=0)
    train_batch_sampler = SimpleBatchSampler(sampler=train_sampler)
    val_sampler = HighThroughputSampler(valset, shuffle_batches=False, roll=True, max_frames=args.max_frame_cache,
                                        debug=0)
    val_batch_sampler = SimpleBatchSampler(sampler=val_sampler)
    train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=1,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(valset, batch_sampler=val_batch_sampler, num_workers=1,
                                             collate_fn=collate_fn)

    # Define network, optimizer and criterion
    if args.concatenation == True:
        inp_size = [np.sum(trainset.channel_dimensions)]
        args.num_sensors = 1
    else:
        inp_size = trainset.channel_dimensions

    net = slx.audio_stan(num_sensors=args.num_sensors, inp_size=inp_size, tra_size=args.tra_size,
                         att_size=args.att_size, att_share=args.att_share, cla_size=args.cla_size, cla_layers=args.cla_layers,
                         num_classes=args.classes,
                         tra_type=args.tra_type, rnn_mode=args.rnn_type, cla_dropout=args.cla_dropout)

    if args.load is not None:
        # Load network
        basedir = '/Data/Dropbox/PhD/Projects/caesar_iscas2019/pytorch/models/'
        net.load_state_dict(torch.load(basedir + args.load + '/best', map_location=lambda storage, loc: storage))

    if args.cuda == True:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), weight_decay=args.weight_decay)
    criterion = CTCLoss()
    print(net)

    # Print parameter count
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        params += sizes
    print('::: # network parameters: ' + str(params))

    # Weight initialization --> accelerate training with Xavier
    dict = {}  # we can store the weights in this dict for convenience
    for name, param in net.named_parameters():
        if 'weight' in name:  # all weights
            weight_init.xavier_uniform(param, gain=1.6)
            if args.rnn_type == 'SRU':
                print('SRU mode')
                weight_init.uniform(param, -0.05, 0.05)
            dict[name] = param
        if 'bias' in name:  # all biases
            weight_init.constant(param, 0)
        if args.rnn_type == 'LSTM':  # only LSTM biases
            if ('bias_ih' in name) or ('bias_hh' in name):
                no4 = int(len(param) / 4)
                no2 = int(len(param) / 2)
                weight_init.constant(param, 0)
                weight_init.constant(param[no4:no2], 1)

    # Get progress tracker, validation meter
    prog_track = progress.progress_tracker(wait_period=args.wait_period, max_patience=args.patience)
    val_meter = metric.meter(blank=0)

    # Epoch loop
    print("Starting training...")
    t_step = 0
    v_step = 0

    # Save pretrained net
    torch.save(net.state_dict(), os.path.join(savedir, 'pretrain'))
    torch.save(net, os.path.join(savedir, 'pretrain_mdl'))

    for epoch in range(args.num_epochs + 1):

        train_loss = 0
        train_batches = 0
        tstart = timer()

        # Batch loop - training
        for sensor_list, feature_length, label, label_length, debug in tqdm(train_loader):
            if train_batches == 0:
                print(sensor_list[0].size()[0] * sensor_list[0].size()[1], label_length.numpy()[:10])
            # Pytorch cuda cached memory allocator: avoid reallocating by making the first batch the biggest
            if t_step == 0:
                sensor_list = max_frame_batch(sensor_list, args.max_frame_cache * 2.1)

            # Convert data to Autograd Variables
            if args.cuda == True:
                sensor_list = [Variable(sensor).cuda() for sensor in sensor_list]
            else:
                sensor_list = [Variable(sensor) for sensor in sensor_list]
            feature_length, label, label_length = Variable(feature_length), Variable(label), Variable(label_length)

            # Optimization
            optimizer.zero_grad()
            output, debug_dict = net(sensor_list, feature_length.data.numpy())
            loss = criterion(output, label, feature_length, label_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 20)
            optimizer.step()

            # Increment monitoring variables
            batch_loss = loss.data.numpy()[0]
            train_loss += batch_loss  # Accumulate loss
            train_batches += 1  # Accumulate count so we can calculate mean later
            t_step += 1

            # Garbage collection to free VRAM
            del sensor_list[:], loss, output, feature_length, label, label_length, debug_dict

            # Write to tensorboard
            logger.scalar_summary('Training-loss over steps', batch_loss, t_step)

        tend = timer()
        logger.scalar_summary('Training-loss over epochs', train_loss / train_batches, epoch)

        # Validation
        # if epoch % np.floor(0.1 * args.num_epochs) == 0:  # validation is done only every num_epochs/10th epoch
        if epoch < 1000:
            val_loss = 0
            val_batches = 0
            vstart = timer()


            # Batch loop - validation
            for sensor_list, feature_length, label, label_length, debug in val_loader:
                # print(sensor_list[0].size()[0] * sensor_list[0].size()[1], label_length.numpy()[:10])
                # Convert data to Autograd Variables on cuda
                sensor_list = [Variable(sensor, volatile=True) for sensor in sensor_list]
                if args.cuda == True:
                    sensor_list = [sensor.cuda() for sensor in sensor_list]
                feature_length, label, label_length = Variable(feature_length, volatile=True), \
                                                      Variable(label, volatile=True), \
                                                      Variable(label_length, volatile=True)

                # Test step
                output, debug_dict = net(sensor_list, feature_length.data.numpy())
                loss = criterion(output, label, feature_length, label_length)

                # Tensorboard default logs
                if val_batches in [0]:
                    print(sensor_list[0].size(), label_length.data.numpy()[:10])

                    plot_idx = 0
                    debug_dict = net.debug_to_numpy(debug_dict)
                    info = {
                        # 'merge': plot_image(debug_dict['merge'][plot_idx].T, title='Merge'),
                        'GRU output': plot_image(debug_dict['classifier'][plot_idx].T, title='Classifier'),
                        'dense output': plot_image(debug_dict['dense'][plot_idx].T, title='Output'),
                    }
                    for sensor in range(net.num_sensors):
                        info['input_{}'.format(sensor)] = plot_image(debug_dict['inputs'][sensor][plot_idx].T,
                                                                     title='sensor_{}'.format(sensor))
                        # info['scale_{}'.format(sensor)] = plot_image(debug_dict['scale'][sensor][plot_idx].T,
                        #                                              title='scaled_transform_{}'.format(sensor))
                        info['transform_{}'.format(sensor)] = plot_image(
                            debug_dict['transforms'][sensor][plot_idx].T,
                            title='transform_{}'.format(sensor))

                    for tag, images in info.items():
                        logger.image_summary(tag, images, epoch)

                # Noise level / attention plots
                # if val_batches in [0]:
                #     batch_size = len(feature_length)
                #     plot_idx = list(np.linspace(0, batch_size - 1, 4, dtype=int))
                #     noise_levels = debug['sensor_noise']
                #     info = {'sm_attentions{}'.format(val_batches): multi_plot_an(debug_dict['sm_attentions'],
                #                                                                  noise_levels, plot_idx)}
                #     for tag, images in info.items():
                #         logger.image_summary(tag, images, epoch)

                # Get prediction and best path decoding
                val_meter.extend_guessed_labels(output.cpu().data.numpy())
                val_meter.extend_target_labels(bY=label.data.numpy(), b_lenY=label_length.data.numpy())

                # Increment monitoring variables
                val_loss += loss.data.numpy()[0]
                val_batches += 1  # Accumulate count so we can calculate mean later
                v_step += 1

                # Delete variables
                del sensor_list[:], loss, output, feature_length, label, label_length, debug_dict

            PER, WER, CER = val_meter.get_metrics()
            prog_track.update(CER)

            logger.scalar_summary('Validation-WER over epochs', WER, epoch)
            logger.scalar_summary('Validation-CER over epochs', CER, epoch)
            logger.scalar_summary('Validation-loss over epochs', val_loss / val_batches, epoch)
            vend = timer()

            print(
                "PID {:5} ::: Epoch {:2} of {:2} ::: Train Time {:5.2f} ::: Val Time {:5.2f} ::: Training Loss {:5.2f} ::: Validation Loss {:5.2f} ::: Val "
                "PER {:.3f} ::: Val WER {:.3f} ::: Val CER {:.3f} ::: Computed On {} Batches" \
                    .format(os.getpid(), epoch, args.num_epochs, tend - tstart, vend - vstart,
                            train_loss / train_batches,
                            val_loss / val_batches,
                            PER, WER, CER,
                            val_batches))
            log_dict['epoch'] = epoch
            log_dict['train_loss'] = train_loss / train_batches
            log_dict['val_loss'] = val_loss / val_batches
            log_dict['val_PER'] = PER
            log_dict['val_WER'] = WER
            log_dict['val_CER'] = CER
            log_dict['#Parameters'] = params
            for key, value in vars(args).items():
                log_dict[key] = value
            log.write_log(logfile, log_dict)

            # Save if best epoch
            if prog_track.best_bool == True:
                torch.save(net.state_dict(), os.path.join(savedir, 'best'))
                torch.save(net, os.path.join(savedir, 'best_mdl'))
                print('>>> saving best model from epoch {}'.format(epoch))

        # Save result
        torch.save(net.state_dict(), os.path.join(savedir, 'recent_{}'.format(epoch)))
        torch.save(net, os.path.join(savedir, 'recent_mdl_{}'.format(epoch)))

        # Check patience
        if prog_track.break_bool == True:
            print('>>> training finished with patience {}'.format(prog_track.patience))
            break

    print('Completed.')
