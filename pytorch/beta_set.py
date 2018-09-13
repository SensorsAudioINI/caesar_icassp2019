import copy
import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import tables
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

class dset(Dataset):
    def __init__(self, h5file, dataset_mode, subset, speaker_wise=False, transform=None):
        # Parameters
        self.h5file = h5file
        self.dataset_mode = dataset_mode
        self.subset = subset
        self.transform = transform
        self.speaker_wise = speaker_wise

        self.h5 = tables.open_file(h5file, 'r')
        self.dataset = self.h5.get_node(os.path.join(os.sep, self.dataset_mode, self.subset))

        # features and channels
        self.channels = self.h5.list_nodes(os.path.join(os.sep, self.dataset_mode, self.subset))
        self.lead = self.channels[0]
        self.channel_dimensions = [int(channel.features.attrs.feature_dimensionality) for channel in self.channels]

    def __len__(self):
        return len(self.lead.key)

    def __getitem__(self, idx):
        '''
        Get a single item of /mode/subset

        Welcome to hell: a sample *CAN* have multiple channels! Thusly, a sample is defined as a list of channels. Each
        channel has its own dictionary, with some information being redundant across channels --> minor overhead and
        better for freakin' debuggin' in case you are whining around. Redundant information is removed by collate_fn
        at a later point.

        Transformations act on a whole sample.

        :param subset_idx: index of a sample in the subsets
        :return:
        '''
        sample = []

        for ch in self.channels:
            attributes = ch.features.attrs

            channel_dict = {}  # new feature dict for every sample

            # fill the dictionary
            channel_dict['features'] = np.reshape(ch.features[idx], ch.feature_shape[idx])
            channel_dict['feature_shape'] = ch.feature_shape[idx]
            channel_dict['label'] = ch.label[idx]
            channel_dict['label_length'] = ch.label_length[idx]
            channel_dict['label_string'] = ch.label_string[idx]
            channel_dict['mode'] = self.dataset_mode
            channel_dict['channel'] = attributes.channel
            channel_dict['modality'] = attributes.modality.decode('utf-8')
            channel_dict['key'] = ch.key[idx]
            channel_dict['sample'] = idx
            channel_dict['mean'] = attributes.mean
            channel_dict['std'] = attributes.std

            if self.speaker_wise == True:
                channel_dict['mean'], channel_dict['std'] = self.chime_speaker_wise(ch, channel_dict['key'])

            sample.append(channel_dict)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def chime_speaker_wise(self, channel, key):
        speaker = key[0].split('_')[0]

        path = os.path.join(channel._v_pathname, 'speaker_wise', speaker)
        node = self.h5.get_node(path)

        return node._v_attrs['mean'], node._v_attrs['std']

def collate_fn(batch):
    "Pad samples, stack and collect in list"

    # Parameters
    debug_keys = ['label', 'label_length', 'label_string', 'sample', 'mode', 'key', 'modality', 'channel']

    # 1. Get max sequence length, number of channels, number of samples
    sequence_length = [sample[0]['feature_shape'][0] for sample in batch]  # Get sequence lengths
    max_length = max(sequence_length)  # Get max sequence length
    num_channels = len(batch[0])
    num_samples = len(batch)

    # Sort from longest to shortest sample (preparation for packed sequence)
    sort_by_length = np.argsort(sequence_length)[::-1]
    batch = [batch[idx] for idx in sort_by_length]
    sequence_length = [sample[0]['feature_shape'][0] for sample in batch]  # Update sequence lengths

    # 2. Pad'n'stack samples in list of tensors
    sensor_list = []
    label = []
    label_length = []
    feature_length = []
    debug = {'sensor_noise': [], 'debug': []}

    for channel in range(num_channels):

        tensor_list = []
        sample_noise = []

        for sample in range(num_samples):

            # Padding
            feature_dict = batch[sample][channel]
            npad = max_length - feature_dict['feature_shape'][0]
            padded_features = np.pad(feature_dict['features'], [(0, npad), (0, 0)], mode='constant')
            tensor = torch.from_numpy(padded_features)
            tensor_list.append(tensor)

            # STAN exclusive: get noise levels
            if 'noise_levels' in feature_dict.keys():
                padded_noise_level = np.pad(feature_dict['noise_levels'], [(0, npad), (0, 0)], mode='constant')
                sample_noise.append(padded_noise_level)
            else:
                sample_noise.append(np.zeros((max_length, 1)))
            # Only needed for one sample
            if channel == 0:
                feature_length.append(feature_dict['feature_shape'][0])
                label.extend(feature_dict['label'])
                label_length.append(feature_dict['label_length'][0])
                debug_dict = {allowed_key: feature_dict[allowed_key] for allowed_key in debug_keys}
                debug['debug'].append(debug_dict)
        tensor_stack = torch.stack(tensor_list, 0)
        sensor_list.append(tensor_stack)

        feature_length = np.asarray(feature_length)
        label = np.asarray(label)
        label_length = np.asarray(label_length)

        # STAN exclusive
        noise_stack = np.stack(sample_noise)
        debug['sensor_noise'].append(noise_stack)

    return sensor_list, torch.from_numpy(feature_length), torch.from_numpy(label), torch.from_numpy(label_length), debug


class HighThroughputSampler(Sampler):
    def __init__(self, data_source, max_frames, shuffle_batches=False, num_splits=1, roll=False, debug=0):
        self.data_source = data_source
        self.shuffle_batches = shuffle_batches
        self.num_splits = num_splits
        self.max_frames = max_frames
        self.debug = debug
        self.roll = roll
        self.length = len(self.data_source)

    def __iter__(self):

        # Get feature shapes of first channel (lead channel)
        feature_shape = self.data_source.lead.feature_shape

        # Zip samples and length
        self.unsorted_index_length = [(index, shape[0]) for index, shape in enumerate(feature_shape)]

        # Sort by length
        # self.index_length = self.unsorted_index_length # only for debugging purposes

        # Split into subsets that are ordered by length --> higher batch variability if num_splits > 1
        self.index_length = self.split_shuffle()

        # Roll array by half of length (useful for validation to get medium-length sequences in the first batch)
        if self.roll == True:
            self.index_length = np.roll(self.index_length, int(len(self.index_length) / 2), axis=0)

        # Max frame cache - group into batches already
        batches = self.max_frame_cache()
        self.length = len(batches)

        # Shuffle the order of batches
        if self.shuffle_batches == True:
            np.random.shuffle(batches)

        # Debugging - deactivated by default
        if self.debug == 1:
            print[batches[0]]
            length_unsorted = np.asarray([element[1] for element in self.unsorted_index_length])

            # print('First 10 indices: {}'.format(index[:10]))
            frames = np.concatenate([length_unsorted[batch] for batch in batches])
            frames_share = [np.sum(length_unsorted[batch]) / float(self.max_frames) for batch in batches]
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(frames, '.')
            ax1.grid()
            ax1.set_title('Sequence lengths')
            ax2.plot(frames_share, '.')
            ax2.grid()
            ax2.set_title('Non-zero share of frame cache \n mean {:.2f} ::: std {:.2f} ::: min {:.2f} ::: max {:.2f}'
                          .format(np.mean(frames_share), np.std(frames_share), np.min(frames_share),
                                  np.max(frames_share)))
            plt.tight_layout()
            plt.show()

        return iter(batches)

    def __len__(self):
        return self.length

    def split_shuffle(self):
        # Randomize order on a cloned array
        clone = copy.deepcopy(self.unsorted_index_length)
        np.random.shuffle(clone)

        # Split and order by length (if num_splits == 1 , just order by length)
        all_sorted_splits = []
        all_splits = np.array_split(clone, self.num_splits)
        for unsorted_split in all_splits:
            split = np.asarray(sorted(unsorted_split, key=lambda x: (x[1], x[0])))
            all_sorted_splits.append(split)

        index_length = np.concatenate(all_sorted_splits)

        return index_length

    def max_frame_cache(self):
        batches = []
        mini_batch = []
        max_len = 0

        for index, length in self.index_length:
            # get total frames if new sample was added wrt padding

            max_len = max(length, max_len)
            total_frames = (len(mini_batch) + 1) * max_len
            # Decide if new sample is added
            if total_frames <= self.max_frames:
                mini_batch.append(index)
            else:

                # Pycharm Debugging helper variables
                # mbatch_lens=feature_lens[mini_batch]
                # dec = np.max(mbatch_lens)
                # mbatch_length = len(mini_batch)
                # dbg=len(mini_batch)*np.max(feature_lens[mini_batch])

                batches.append(mini_batch)
                mini_batch = [index]
                max_len = length

        batches.append(mini_batch)
        return batches


class SimpleBatchSampler(object):
    """A sampler that yields received batches

    This sampler takes a list of batches and yields one batch after the other.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        for batch in self.sampler:
            yield batch

    def __len__(self):
        return len(self.sampler)


def max_frame_batch(sensor_list, max_frames):
    expanded = []
    for sensor in sensor_list:
        b, t, f = sensor.size()
        frames = b * t
        diff = max_frames - frames
        pad = int(diff / b + 1)

        embed = torch.zeros(b, t + pad, f)
        embed[:, :t, :] = sensor
        expanded.append(embed)

    return expanded


if __name__ == '__main__':
    file = '/media/stefbraun/ext4/temp/test.h5'

    ds = dset(h5file=file, dataset_mode='isolated', subset='et05_real')
    test = ds.__len__
    start = timer()
    for element in tqdm(ds):
        5 + 5
        # print(element)
    end = timer()
    print(end - start)
    5 + 5
