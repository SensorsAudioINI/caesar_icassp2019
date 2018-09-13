import matplotlib

# matplotlib.use('Agg')  # from https://stackoverflow.com/questions/27147300/how-to-clean-images-in-python-django
fontsize = 18
font = {'size': fontsize}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def get_figure():
    # https://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots
    fig = plt.figure(num=0, figsize=(6, 4), dpi=300)
    fig.clf()
    return fig


def fig2rgb_array(fig, expand=True):
    # https://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


# def plot_image(image, title='Title'):
#     fontsize = 20
#     font = {'size': fontsize}
#     matplotlib.rc('font', **font)
#
#     mean = 'N/A'
#     std = 'N/A'
#     try:
#         mean = np.mean(image)
#         std = np.std(image)
#     except:
#         pass
#     fig, ax = plt.subplots(figsize=(10.5, 4.5))
#     im1 = ax.imshow(image, aspect='auto', interpolation='none', cmap='viridis')
#
#     fig.colorbar(im1, ax=ax)
#     ax.set_title('{} - mean:{:.4f} - std :{:.4f}'.format(title, mean,std), fontsize=fontsize)
#     ax.set_ylabel('channels')
#     ax.set_xlabel('frames')
#     plt.tight_layout()
#     num = fig2rgb_array(fig)
#     plt.close()
#     return num

def plot_image(image, title='Title', figsize=(10.5, 4.5)):
    mean = 'N/A'
    std = 'N/A'
    try:
        mean = np.mean(image)
        std = np.std(image)
    except:
        pass
    fig, ax = plt.subplots(figsize=figsize)

    im1 = ax.imshow(image, aspect='auto', interpolation='none', cmap='viridis')
    ax.locator_params(nbins=3, axis='x')
    ax.locator_params(nbins=3, axis='y')

    cb = fig.colorbar(im1, ax=ax, orientation='vertical')
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()

    ax.set_title('{} - mean:{:.4f} - std :{:.4f}'.format(title, mean, std))
    ax.set_ylabel('channels')
    ax.set_xlabel('frames')

    plt.tight_layout()
    num = fig2rgb_array(fig)
    plt.close()
    return num


def plot_an(sm_attentions, noise_levels, plot_idx, force_plot=0):
    # Get labels and parameters
    labels = ['sensor_{}'.format(i) for i in range(len(sm_attentions))]

    sm_attentions = np.concatenate(sm_attentions, axis=2)
    noise_levels = np.concatenate(noise_levels, axis=2)

    max_frames = sm_attentions.shape[1]

    # Get plot and color cycle
    fig, (ax0, ax1) = plt.subplots(2, figsize=(10.5, 4.5), sharex=True)
    color_cycle = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    ax0.set_prop_cycle('color', color_cycle)
    ax1.set_prop_cycle('color', color_cycle)

    # Plot noise levels
    ax0.plot(noise_levels[plot_idx, :, :])
    ax0.grid()
    ax0.set_title('Noise levels')
    ax0.set_xlim((0, max_frames))
    ax0.legend(labels, loc=1, fontsize=10)
    ax0.set_ylabel('$\sigma$')
    ax0.legend(labels, loc=1, fontsize=10)
    ax0.set_ylim(bottom=0)

    # Plot attentions
    ax1.plot(sm_attentions[plot_idx, :, :])
    ax1.set_ylim((0, 1))
    ax1.grid()
    ax1.set_title('Attention')
    # ax1.set_xlabel('Frames')


    plt.tight_layout()
    if force_plot == 0:
        num = fig2rgb_array(fig)
        plt.close()  # TODO uncomment for training
        return num

def plot_a(ax,sm_attentions, plot_idx):
    # Get labels and parameters
    labels = ['ch{}'.format(i) for i in range(1,len(sm_attentions)+1)]

    sm_attentions = np.concatenate(sm_attentions, axis=2)

    max_frames = sm_attentions.shape[1]

    # Get plot and color cycle
    # color_cycle = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    # ax.set_prop_cycle('color', color_cycle)

    # Plot attentions
    ax.plot(sm_attentions[plot_idx, :, :], linewidth=2)
    ax.set_ylim((0, 1))
    ax.grid(color='k', linestyle='--')
    # ax.set_title('Attention')
    # ax.set_xlabel('Frames')
    # ax.legend(labels, loc=1, fontsize=10, ncol=6)

    return ax

def multi_plot_an(sm_attentions, noise_levels, plot_idx=[0,1,2,3]):
    num_list = []
    for idx in plot_idx:
        num_list.append(plot_an(sm_attentions, noise_levels, idx))
    return np.asarray(num_list)[:, 0, :, :, :]


def to_np(x):
    return x.data.cpu().numpy()
