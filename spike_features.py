import numpy as np

def spike_count_features_by_time(timestamps, addresses, twl=0.005, tws=0.005, nb_channels=64, **options):
    """Creates spike count features for an audio event stream, by time binning.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param twl: The length of time window for bunching the events.
        :param tws: The shift of time window in bunching the events.
        :param nb_channels: The number of frequency channels.
    Returns:
        The spike count features for the event_stream; of shape (nb_time_bins, nb_channels).
    """
    time_in_stream = timestamps[-1] - timestamps[0]
    nb_bins = int(np.ceil(time_in_stream / tws))
    data_array_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    current_time = timestamps[0]
    frame = 0
    while True:
        indices_left = current_time <= timestamps
        indices_right = timestamps < current_time + twl
        indices = np.multiply(indices_left, indices_right)
        if np.amax(indices):
            current_addresses = addresses[indices]
            data_array_to_return[frame] = np.histogram(current_addresses, bins=range(nb_channels + 1))[0] \
                .astype(np.float32, copy=False)
        frame += 1
        current_time += tws
        if frame == nb_bins:
            break
    return data_array_to_return


def exponential_features_by_time(timestamps, addresses, twl=0.005, tws=0.005, nb_channels=64, bunching='average',
                                 tau_type='constant', **options):
    """Creates time bunched exponential features for an audio event stream.
    The function first creates the exponential features for the events in the event stream through the function
    exponential_features. Then the events in every time bin are bunched together and the time bunched
    feature is created through either averaging or summing the features together.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param twl: The length of time window for bunching the events.
        :param tws: The shift of time window in bunching the events.
        :param nb_channels: The number of frequency channels.
        :param bunching: The mode of bunching the events. If 'average', the features for the events in each time bin are
        averaged, while if 'sum', the features for the events in each time bin are summed.
        :param tau_type: The type of tau to be used for the features.
    Returns:
        The time bunched exponential features for the event stream; of shape (nb_time_bins, nb_channels).
    """
    time_in_stream = np.amax(timestamps) - np.amin(timestamps)
    nb_bins = int(np.ceil(time_in_stream / tws))
    features_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    if tau_type == 'constant':
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    else:
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    current_time = timestamps[0]
    frame = 0
    while True:
        indices_left = current_time <= timestamps
        indices_right = timestamps < current_time + twl
        indices = np.multiply(indices_left, indices_right)
        if np.amax(indices):
            current_events = exp_data[indices]
            if bunching == 'average':
                features_to_return[frame] = np.mean(current_events, axis=0)
            elif bunching == 'sum':
                features_to_return[frame] = np.sum(current_events, axis=0)
        else:
            features_to_return[frame] = np.zeros(shape=nb_channels, dtype='float32')
        frame += 1
        current_time += tws
        if frame == nb_bins:
            break
    return features_to_return