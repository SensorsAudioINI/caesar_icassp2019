import numpy as np
import mkl_random as rnd
import copy
import noise_lib as nl
from sklearn import preprocessing
import skvideo.utils as su

class warp_ctc_shift(object):
    def __init__(self, shift=1):
        self.shift = shift

    def __call__(self, sample):
        for channel in sample:
            channel['label'] += 1
        return sample


class standardization():
    def __init__(self, mode):

        if mode not in ['epoch', 'sample']:
            raise NotImplementedError('{} >>> this standardization mode is not implemented.'.format(mode))
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'epoch':

            for idx, channel in enumerate(sample):
                # print channel['mean'].shape
                channel['features'] -= channel['mean']
                channel['features'] /= channel['std']

        elif self.mode == 'sample':
            for channel in sample:
                if channel['modality'] == 'audio':  # Frames x Features
                    channel['features'] = preprocessing.scale(channel['features'], axis=0)
                elif channel['modality'] == 'video':  # Frames X Height X Width X Color channels
                    channel['features'] = channel['features'].astype(np.float32)
                    for color in range(channel['features'].shape[-1]):
                        channel['features'][:, :, :, color] -= channel['features'][:, :, :, color].mean()
                        channel['features'][:, :, :, color] /= (channel['features'][:, :, :, color].std() + 1e-8)

        return sample


class channel_cloner():
    def __init__(self, clone_factor):
        self.clone_factor = clone_factor

    def __call__(self, sample):
        clone_list = []
        for idx in range(self.clone_factor):
            dict_clone = copy.deepcopy(sample[0])
            dict_clone['channel'] = idx
            clone_list.append(dict_clone)

        return clone_list


class random_walk_noise():
    def __init__(self, random=True, max_lim=3, sudden_death=0):
        self.random = random
        self.max_lim = max_lim + 1e-8
        self.counter = 0
        self.sudden_death = sudden_death

    def __call__(self, sample):

        for channel in sample:
            if self.random == True:
                noise, noise_levels = nl.random_walk_noise(np.expand_dims(channel['features'], axis=0), debug=1,
                                                           max_lim=self.max_lim)
            if self.random == False:
                noise, noise_levels = nl.random_walk_noise(np.expand_dims(channel['features'], axis=0), epoch=1,
                                                           sensor=channel['channel'], debug=1, max_lim=self.max_lim)
            if self.random == 'counter':
                noise, noise_levels = nl.random_walk_noise(np.expand_dims(channel['features'], axis=0),
                                                           epoch=self.counter,
                                                           sensor=channel['channel'], debug=1, max_lim=self.max_lim)
                self.counter += 1

            if self.sudden_death > 0:  # Get parameters
                max_len, channels = channel['features'].shape
                # Create mask for batch
                mask = np.less(noise_levels, self.sudden_death)
                mask = np.repeat(mask, channels, axis=2)

                # Fill zeros with random normal noise with zero mean and unit variance
                fallback_batch = np.invert(mask) * np.random.normal(loc=0.0, scale=1.0, size=(1, max_len, channels))
                fallback_batch = fallback_batch.astype(np.float32)

                # Mix it!
                channel['features'] = channel['features'] * np.squeeze(mask) + np.squeeze(fallback_batch + noise)
                channel['noise_levels'] = np.squeeze(noise_levels, axis=0)
            else:
                channel['features'] += np.squeeze(noise)
                channel['noise_levels'] = np.squeeze(noise_levels, axis=0)
        return sample


class symmetric_random_walk():
    def __init__(self, random=True, scale=3.0, normalize=True, sudden_death=0):

        # Parameters
        self.random = random
        self.scale = scale + 1e-8
        self.normalize = normalize
        self.sudden_death = sudden_death

        # Random seed counter
        self.counter = 0

    def __call__(self, sample):

        for channel in sample:
            if self.random == True:
                noise, noise_levels = nl.symmetric_random_walk(channel['features'].shape, scale=self.scale,
                                                               normalize=self.normalize)

            if self.random == False:
                noise, noise_levels = nl.symmetric_random_walk(channel['features'].shape, seed=channel['channel'],
                                                               scale=self.scale,
                                                               normalize=self.normalize)

            if self.random == 'counter':
                noise, noise_levels = nl.symmetric_random_walk(channel['features'].shape,
                                                               seed=channel['channel'] + self.counter, scale=self.scale,
                                                               normalize=self.normalize)
                self.counter += 1

            if self.sudden_death > 0:  # Get parameters
                max_len, channels = channel['features'].shape
                # Create mask for batch
                mask = np.less(noise_levels, self.sudden_death)
                mask = np.repeat(mask, channels, axis=2)

                # Fill zeros with random normal noise with zero mean and unit variance
                fallback_batch = np.invert(mask) * np.random.normal(loc=0.0, scale=1.0, size=(1, max_len, channels))
                fallback_batch = fallback_batch.astype(np.float32)

                # Mix it!
                channel['features'] = channel['features'] * np.squeeze(mask) + np.squeeze(fallback_batch + noise)
                channel['noise_levels'] = np.squeeze(noise_levels, axis=0)
            else:
                channel['features'] += noise
                channel['noise_levels'] = noise_levels

        return sample


class cross_noise():
    def __init__(self, max_lim=3):
        self.max_lim = max_lim

    def __call__(self, sample):
        for channel in sample:
            if channel['channel'] % 2 == 0:
                start_lvl = 0
                end_lvl = self.max_lim
            else:
                start_lvl = self.max_lim
                end_lvl = 0
            noise, noise_levels = nl.increasing_noise(np.expand_dims(channel['features'], axis=0), start_lvl=start_lvl,
                                                      end_lvl=end_lvl, debug=1)
            noise_levels = np.expand_dims(noise_levels, 0)
            noise_levels = np.expand_dims(noise_levels, 2)

            channel['features'] += np.squeeze(noise)
            channel['noise_levels'] = np.squeeze(noise_levels, axis=0)
        return sample


class laplacian_noise():
    def __init__(self, sigma=0.6):
        self.sigma = sigma + 1e-8

    def __call__(self, sample):
        for channel in sample:
            noise = rnd.normal(loc=0.0, scale=self.sigma, size=channel['features'].shape).astype(np.float32)
            channel['features'] += noise
        return sample


class gaussian_noise():
    def __init__(self, sigma=0.6):
        self.sigma = sigma + 1e-8

    def __call__(self, sample):
        for channel in sample:
            noise = rnd.normal(loc=0.0, scale=self.sigma, size=channel['features'].shape).astype(np.float32)
            channel['features'] += noise
        return sample


class time_series_2d():
    def __init__(self):
        pass

    def __call__(self, sample):
        for channel in sample:
            if channel['modality'] == 'video':  # Frames X Height X Width X Color channels
                frames = channel['features'].shape[0]
                channel['features'] = np.reshape(channel['features'], (frames, -1))
        return sample


class rgb2gray():
    def __init__(self):
        pass

    def __call__(self, sample):
        for channel in sample:
            if channel['modality'] == 'video':
                channel['features'] = su.rgb2gray(channel['features'])

        return sample


class concatenation():
    def __init__(self):
        pass

    def __call__(self, sample):
        stack = []
        for channel in sample:
            stack.append(channel['features'])

        stacked = np.concatenate(stack, axis=1)

        stacked_sample = [channel]
        stacked_sample[0]['features'] = stacked

        return stacked_sample


class length_equalizer():
    def __init__(self):
        pass

    def __call__(self, sample):
        # Get min length of all channels
        min_len = [np.Inf]
        for channel in sample:
            min_len = min(min_len, channel['feature_shape'][0])
        # Cut to length
        for channel in sample:
            channel['features'] = channel['features'][:min_len]
            channel['feature_shape'][0] = min_len

        return sample


class keep_channel():
    def __init__(self, keep_channel):
        self.keep_channel = keep_channel

    def __call__(self, sample):
        sample = [sample[self.keep_channel]]
        return sample


class disable():
    def __init__(self, disabled, distype='null'):
        self.disabled = disabled
        self.distype = distype

    def __call__(self, sample):
        for idx, channel in enumerate(sample):
            if idx in self.disabled:

                if self.distype == 'null':
                    channel['features'] *= 0

                if self.distype == 'gaussian':
                    channel['features'] = np.random.normal(size=channel['features'].shape).astype(np.float32)
        return sample