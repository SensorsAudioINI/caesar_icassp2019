import numpy as np

import torch
from torch.autograd import Variable
import stan_lib as sl

import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)

if __name__ == '__main__':

    np.random.seed(123)

    sensor = np.random.random((3,20,39)).astype(np.float32)
    sensor_var = Variable(torch.from_numpy(sensor))
    stan = sl.audio_stan(num_sensors=2, inp_size=[39,39], tra_size=50, att_size=20, cla_size=200, classes=12, \
                         dense_activation='selu', rnn_activation='tanh')
    print(stan)
    result = stan([sensor_var, sensor_var*0])
    stan.debug_to_numpy(stan.merge)

    # plot
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(16, 9))
    ax1.plot(stan.debug_numpy['sm_attentions'][0][0, :, :])
    ax1.set_title('Attention 0')
    ax1.grid()
    ax2.plot(stan.debug_numpy['sm_attentions'][1][1, :, :])
    ax2.grid()
    ax2.set_title('Attention 1')
    ax3.imshow(stan.debug_numpy['scale'][0][0, :, :].T, aspect='auto')
    ax3.set_title('Scale 0')
    ax4.imshow(stan.debug_numpy['scale'][1][0, :, :].T, aspect='auto')
    ax4.set_title('Scale 1')
    ax5.imshow(stan.debug_numpy['merge'][0, :, :].T, aspect='auto')
    ax5.set_title('Merge')
    # numpy verification of sensor merge layer
    np_attended = np.concatenate(stan.debug_numpy['attention'], axis=-1)
    np_softmax_attended = softmax(np_attended)
    np_scaled = []
    for idx in range(stan.num_sensors):
        np_scaled.append(np.expand_dims(np_softmax_attended[:, :, idx], axis=-1) * stan.debug_numpy['transforms'][idx])
    np_merged = np.sum(np_scaled, axis=0)

    im6 = ax6.imshow(stan.debug_numpy['merge'][0, :, :].T - np_merged[0, :, :].T, aspect='auto')
    ax6.set_title('Difference numpy vs tensorflow')
    plt.colorbar(im6, ax=ax6)
    print('Check if merged arrays are close: {}'.format(np.allclose(np_merged, stan.debug_numpy['merge'], atol=1e-6)))
    print('Check if softmax of attention sums up to 1: {}'.format(
        np.allclose(np.sum(np.concatenate(stan.debug_numpy['sm_attentions'], axis=-1), axis=2), 1)))
    plt.tight_layout()
    plt.show()