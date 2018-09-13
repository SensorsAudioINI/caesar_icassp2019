import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(transformation_module, self).__init__()
        self.fc = nn.Linear(in_features=inp_size, out_features=tra_size)


    def forward(self, x):

        out = self.fc(x)
        out = F.selu(out)

        return out


class identity_module(nn.Module):
    def __init__(self):
        super(identity_module, self).__init__()

    def forward(self, x):
        return x

class attention_module(nn.Module):
    def __init__(self, inp_size, att_size):
        super(attention_module, self).__init__()

        # Transformation
        self.gru = nn.GRU(input_size=inp_size, hidden_size=att_size, num_layers=1, bias=True, batch_first=True,
                          bidirectional=False)
        self.fc = nn.Linear(in_features=att_size, out_features=1)

    def forward(self, x, x_len):
        # print(x_len)
        x = pack_padded_sequence(x, x_len, batch_first=True)
        out, _ = self.gru(x)

        # packed sequence speed hack states
        # h1 = self.fc(out.data)
        # h2 = PackedSequence(h1[:,:], out.batch_sizes)
        # out,_ = pad_packed_sequence(h2, batch_first=True)
        #
        # out = F.leaky_relu(out)

        # conventional
        out,_ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)
        out = F.selu(out)
        return out


class merge_module(nn.Module):
    def __init__(self, num_sensors):
        super(merge_module, self).__init__()
        self.num_sensors = num_sensors
        # Sensor merge

    def forward(self, x_attention, x_transform):
        # MINOR MESS WARNING: pytorch does not support dim argument in softmax, hence the transposing.
        softmax_attention = F.softmax(torch.cat(x_attention, -1).transpose(0, 2)).transpose(0, 2)
        split_attention = softmax_attention.split(1, 2)

        scale = []
        for idx in range(self.num_sensors):
            scale.append(split_attention[idx] * x_transform[idx])

        merge_stack = torch.stack(scale)
        merge = merge_stack.sum(0)

        return merge, list(split_attention), scale


class classification_module(nn.Module):
    def __init__(self, inp_size, cla_size, num_classes, num_layers):
        super(classification_module, self).__init__()
        self.gru = nn.GRU(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers, bias=True,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2 * cla_size, out_features=num_classes, bias=True)

    def forward(self, x, x_len):
        x = pack_padded_sequence(x, x_len, batch_first=True)
        rnn, _ = self.gru(x)

        # packed sequence speed hack states
        h1 = self.fc(rnn.data)
        h2 = PackedSequence(h1, rnn.batch_sizes)
        fc, _ = pad_packed_sequence(h2, batch_first=True)

        # usual packed sequence
        # rnn, _ = pad_packed_sequence(rnn, batch_first=True)
        # fc = self.fc(rnn)

        return rnn, fc


class audio_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59, tra_type='dense'):
        super(audio_stan, self).__init__()

        # Sensor parameters
        self.num_sensors = num_sensors
        self.inp_size = inp_size

        # Transformation module parameters
        self.tra_size = tra_size
        self.tra_type = tra_type

        # Attention modules parameters
        self.att_size = att_size

        # Classification module parameters
        self.cla_size = cla_size
        self.cla_layers = cla_layers
        self.classes = num_classes

        # Packed sequence hack
        self.classifier = torch.from_numpy(np.zeros((10, 10, 10)))  # packed sequence speed hack

        # List of transformation modules
        if tra_type == 'dense':
            self.transformation_module = nn.ModuleList()
            for idx in range(self.num_sensors):
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

        if tra_type == 'identity':
            self.transformation_module = nn.ModuleList()
            for idx in range(self.num_sensors):
                self.transformation_module.append(identity_module())
            self.tra_size=self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            self.attention_module.append(attention_module(inp_size=self.tra_size, att_size=self.att_size))
        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes)

    def forward(self, x, x_len):

        # Transformation
        self.transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            self.transformation.append(t_module(sensor))

        # Attention
        self.attention = []
        for trans, a_module in zip(self.transformation, self.attention_module):
            self.attention.append(a_module(trans, x_len))

        # Merge
        self.merge, self.sm_attention, self.scale = self.merge_module(self.attention, self.transformation)

        # Classification
        _, self.dense = self.classification_module(self.merge, x_len)

        self.ctc_out = self.dense.transpose(0, 1)

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        self.debug = {'inputs': x, 'classifier': self.classifier, 'dense': self.dense, 'output': self.ctc_out,
                      'transforms': self.transformation, 'attention': self.attention,
                      'sm_attentions': self.sm_attention,
                      'scale': self.scale, 'merge': self.merge}

        return self.ctc_out

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self):
        self.debug_numpy = {}
        for key, value in self.debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                self.debug_numpy[key] = numpy_list
            else:
                self.debug_numpy[key] = self.to_numpy(value)
