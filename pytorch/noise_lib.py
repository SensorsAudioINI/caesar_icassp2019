import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import mkl_random as rnd

def toy_batch(seed=11, shape=(25, 100, 123)):
    np.random.seed(seed)
    bX = np.float32(np.random.uniform(-1, 1, (shape)))
    bY = np.int32(range(shape[0]))
    b_lenX = [shape[1]] * shape[0]
    maskX = np.float32(np.ones((bX.shape[0], bX.shape[1])))
    return bX, np.asarray(bY), b_lenX, maskX


def increasing_noise(batch, start_lvl, end_lvl, debug=0):
    batch_size, time_steps, feat_dims = batch.shape
    lvls = np.linspace(start_lvl, end_lvl, time_steps)

    noise_sample = np.zeros((time_steps, feat_dims), dtype=np.float32)
    for ts, lvl in enumerate(lvls):
        noise_vec = np.random.normal(0, lvl, feat_dims)
        noise_sample[ts, :] = noise_vec

    noise_sample = normalize_sample(noise_sample)
    noise_batch = np.repeat(noise_sample[np.newaxis, :, :], batch_size, axis=0)

    if debug == 0:
        return noise_batch
    else:
        return noise_batch, lvls


def sinusoidal_noise(batch, debug=0, frequency=None):
    batch_size, time_steps, feat_dims = batch.shape
    x_axis = np.arange(0, time_steps)
    if frequency == None:
        lvls = 1.5 * np.sin(np.random.uniform(0, 1) * x_axis) + 1.5
    else:
        lvls = 1.5 * np.sin(frequency * x_axis) + 1.5

    noise_sample = np.zeros((time_steps, feat_dims), dtype=np.float32)
    for ts, lvl in enumerate(lvls):
        noise_vec = np.random.normal(0, lvl, feat_dims)
        noise_sample[ts, :] = noise_vec
    noise_sample = normalize_sample(noise_sample)
    noise_batch = np.repeat(noise_sample[np.newaxis, :, :], batch_size, axis=0)

    if debug == 0:
        return noise_batch
    else:
        return noise_batch, lvls


def blend_noise(batch, start, end, debug=0):
    batch_size, time_steps, feat_dims = batch.shape
    x_axis = np.arange(0, time_steps)
    lvls = np.zeros(time_steps)
    lvls[start:end] = 3

    noise_sample = np.zeros((time_steps, feat_dims), dtype=np.float32)
    for ts, lvl in enumerate(lvls):
        noise_vec = np.random.normal(0, lvl, feat_dims)
        noise_sample[ts, :] = noise_vec
    blend_sample = normalize_sample(noise_sample[start:end, :])
    noise_sample[start:end, :] = blend_sample
    noise_batch = np.repeat(noise_sample[np.newaxis, :, :], batch_size, axis=0)

    if debug == 0:
        return noise_batch
    else:
        return noise_batch, lvls


def square_noise(batch, frequency=None, shift=0, debug=0):
    batch_size, time_steps, feat_dims = batch.shape
    x_axis = np.arange(0, time_steps + shift)
    if frequency == None:
        lvls = np.sin(np.random.uniform(0, 1) * x_axis) + 1
        lvls[lvls > 1] = 3
        lvls[lvls <= 1] = 0
    else:
        lvls = np.sin(frequency * x_axis) + 1
        lvls[lvls > 1] = 3
        lvls[lvls <= 1] = 0

    lvls = lvls[shift:time_steps + shift]

    noise_sample = np.zeros((time_steps, feat_dims), dtype=np.float32)
    for ts, lvl in enumerate(lvls):
        noise_vec = np.random.normal(0, lvl, feat_dims)
        noise_sample[ts, :] = noise_vec
    noise_sample = normalize_sample(noise_sample)
    noise_batch = np.repeat(noise_sample[np.newaxis, :, :], batch_size, axis=0)

    if debug == 0:
        return noise_batch
    else:
        return noise_batch, np.expand_dims(np.tile(lvls, (batch.shape[0], 1)), 2)


def normalize_sample(sample):
    s_mean = np.mean(sample, axis=0)
    s_std = np.std(sample, axis=0)
    sample = sample - s_mean[np.newaxis, :]
    sample = sample / s_std[np.newaxis, :]
    return sample


def normalize_batch(batch):
    batch_size = batch.shape[0]
    for idx, sample in enumerate(batch):
        s_mean = np.mean(sample, axis=0)
        s_std = np.std(sample, axis=0)
        sample = sample - s_mean[np.newaxis, :]
        sample = sample / s_std[np.newaxis, :]
        batch[idx] = sample
    return batch


def random_walk_noise(batch, epoch=None, sensor=None, scale=0.2, max_lim=3, shape=0.8, thresh=0, debug=0, filter=None,
                      f_start=None,
                      f_stop=None):
    if epoch != None:
        new_seed = int(str(epoch) + str(sensor))
        np.random.seed(new_seed)
    data_size = batch.shape
    batch_size, max_steps = data_size[:2]
    # Generate noise scales using a Gaussian
    noise_signs = np.random.random(size=(batch_size, max_steps)) - 0.5 # sample from uniform distribution in [0,1)
    noise_scales = np.random.gamma(shape=shape, scale=scale, size=(batch_size, max_steps)) * noise_signs / abs(
        noise_signs) # use gamma distribution to get values from [0, inf), then multiply by noise signs

    # Replace the starting point of the walk to be uniform over the range
    noise_scales[:, 0] = np.random.uniform(0, max_lim / 2, size=(batch_size,))
    # Turn into a walk with cumulative sum
    noise_scales = np.abs(np.cumsum(noise_scales, axis=1)).reshape(
        (batch_size, max_steps,) + (1,) * (len(data_size) - 2))

    # Reflect around the boundaries
    # deprecated
    # noise_scales[noise_scales > max_lim] = max_lim * np.ceil(noise_scales[noise_scales > max_lim] / max_lim) - \
    #                                        noise_scales[noise_scales > max_lim]
    # noise_scales[noise_scales < -max_lim] = max_lim * np.ceil(noise_scales[noise_scales < -max_lim] / max_lim) - \
    #                                         noise_scales[noise_scales < -max_lim]

    noise_scales = max_lim - np.abs(np.mod(noise_scales, 2. * max_lim) - max_lim)

    # Threshold
    noise_scales[noise_scales < thresh] = 0

    # Filter
    if filter == True:
        order = 2
        fs = 100
        cutoff = 10
        noise_scales = np.squeeze(noise_scales)
        for idx, noise in enumerate(noise_scales):
            noise_scales[idx, :] = butter_lowpass_filter(noise, cutoff=cutoff, fs=fs, order=order)
        noise_scales = np.expand_dims(noise_scales, axis=2)
        noise_scales[noise_scales < 0] = 0

    # Create noise based on the walk
    # all_noise = np.random.normal(loc=0, scale=noise_scales, size=data_size)
    all_noise = np.random.uniform(low=-np.sqrt(12)*noise_scales/2, high=np.sqrt(12)*noise_scales/2, size=data_size)
    all_noise = all_noise.astype(np.float32)
    if debug == 0:
        return all_noise
    else:
        return all_noise, noise_scales

def symmetric_random_walk(size, seed=None, scale=1.0, normalize=True):
    # 0. Preparation
    if seed != None:
        rnd.seed(seed)
    else:
        rnd.seed()
    time_steps = size[0]
    num_dims = len(size) - 1

    # 1. Generate random walk noise levels (integer)
    random_walk = 2 * rnd.randint(2, size=time_steps) - 1
    random_walk = np.cumsum(random_walk)

    # 2. Normalize random walk  noise levels to the range [0,1]
    if normalize == True:
        random_walk += np.abs(random_walk.min())
        random_walk = random_walk / random_walk.max()

    # 3. Scale random walk noise levels to max_level
    random_walk *= scale

    # 4. Generate noise
    noise = rnd.uniform(low=-np.sqrt(12)/2, high=np.sqrt(12)/2, size=size)

    # 5. Scale noise to desired std over time_steps
    random_walk=random_walk.reshape((-1,) + (1,) * num_dims)
    noise *= random_walk

    return noise.astype(np.float32), random_walk.astype(np.float32)

def multi_style_noise(batch, num_sensor, max_lim=3, debug=1):
    noise, levels = random_walk_noise(batch, max_lim=max_lim, filter=True, debug=debug)

    if num_sensor % 2 == 0:
        sine_noise, sine_level = sinusoidal_noise(batch[2:3, :, :], debug=1, frequency=0.1)
        cross_noise, cross_level = increasing_noise(batch[3:4, :, :], start_lvl=0, end_lvl=3, debug=debug)
    if num_sensor % 2 == 1:
        sine_noise, sine_level = increasing_noise(batch[2:3, :, :], start_lvl=0, end_lvl=1e-8, debug=debug)
        cross_noise, cross_level = increasing_noise(batch[3:4, :, :], start_lvl=3, end_lvl=0, debug=debug)
    noise[2, :, :], levels[2, :, 0] = sine_noise, sine_level
    noise[3, :, :], levels[3, :, 0] = cross_noise, cross_level

    return noise, levels


def plot_first_sample(mode, batch, num_sen, epoch):
    # Decide plot yes/no
    plot_bool = np.zeros(3, dtype=np.int)
    plot_bool[:num_sen] = np.ones(num_sen)

    # Plot and save
    fig, axs = plt.subplots(ncols=1, nrows=3)
    for sensor, ax in enumerate(axs):
        curr_batch = batch[sensor]
        ax.imshow(plot_bool[sensor] * curr_batch[0, :, :].T, aspect='auto')
        ax.set_title(
            'Sensor: {} Mean: {} Std {}'.format(sensor + 1, np.mean(curr_batch[0, :, :]), np.std(curr_batch[0, :, :])))
    plt.tight_layout()
    plt.savefig('figures/{}/e{}.png'.format(mode, epoch))
    plt.close()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decision(probability):
    return np.random.random() < probability


def noise_plot(attentions, noise_levels):
    colors = ['b', 'r', 'g']
    # Attention vs noise
    fig, ax1 = plt.subplots()
    for i, att in enumerate(attentions):
        ax1.plot(att, label='Input {} attention'.format(i), color=colors[i], linewidth=2)
    ax1.set_ylabel('Attention')
    ax1.set_ylim((-0.1, 1.1))
    ax2 = ax1.twinx()
    for i, lvl in noise_levels:
        ax2.plot(lvls1, color=colors[i], ls=':', label='Input {} noise level'.format(i))
    ax2.set_ylabel('Noise level')
    plt.grid()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_ylim((-0.10, 3.1))
    ax2.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)


def sudden_death(batch, noise, noise_levels, thresh=1.0):
    # Get parameters
    batch_size, max_len, channels = batch.shape

    # Create mask for batch
    mask = np.less(noise_levels, thresh)
    mask = np.repeat(mask, channels, axis=2)

    # Fill zeros with random normal noise with zero mean and unit variance
    fallback_batch = np.invert(mask) * np.random.normal(loc=0.0, scale=1.0, size=batch.shape)

    # Mix it!
    masked_mix = batch * mask + fallback_batch + noise

    return masked_mix


if __name__ == '__main__':
    bX, bY, b_lenX, maskX = toy_batch(seed=12)
    tb_noisy, noise_scales = increasing_noise(bX, 1, 1, debug=1)
    tb_noisy = tb_noisy * 3
    print('mean {}'.format(np.mean(tb_noisy)))
    print('std {}'.format(np.std(tb_noisy)))
    # normalize_batch(tb_noisy)
    print(np.mean(tb_noisy))
    print(np.std(tb_noisy))
    lvl = np.squeeze(noise_scales[0, :])
    order = 2
    fs = 100
    cutoff = 10
    lvl_filtered = butter_lowpass_filter(lvl, cutoff=cutoff, fs=fs, order=order)
    plt.figure()
    plt.plot(lvl, color='b')
    # plt.plot(lvl_filtered, color='r')
    plt.grid()
    plt.figure()
    plt.imshow(tb_noisy[1, :, :].T)
    plt.colorbar()
    plt.show()
