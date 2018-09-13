import torch
import torch.nn as nn
import torch.nn.functional as F


# from cuda_functional import SRU

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        new_tuple = (t * n,) + x.size()[2:]

        x_reshape = x.contiguous().view(new_tuple)
        y = self.module(x_reshape)
        # We have to reshape Y

        new_tuple = (t, n) + y.size()[1:]

        y = y.contiguous().view(new_tuple)

        return y


class transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(transformation_module, self).__init__()
        self.fc1 = nn.Linear(in_features=inp_size, out_features=tra_size)
        # self.fc2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.fc3 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        # x = F.selu(self.fc2(x))
        # x = F.selu(self.fc3(x))

        # t, n = out.size(0), out.size(1)
        # new_tuple = (t * n,) + out.size()[2:]
        # out = out.view(new_tuple)

        # out = out.permute(0, 2, 1).contiguous()
        # out = self.bnd(out)
        # out = out.permute(0, 2, 1).contiguous()
        # out = F.selu(out)
        # out = out.view(t, n, 50).contiguous()
        return x


class rnn_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(rnn_transformation_module, self).__init__()
        self.rnn = nn.LSTM(input_size=inp_size, hidden_size=tra_size, num_layers=1, bias=True, bidirectional=False,
                           batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)

        return out


class cnn_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(cnn_transformation_module, self).__init__()

        # big
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
        #                        kernel_size=5,
        #                        stride=1)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=5)
        # self.bn4 = nn.BatchNorm2d(64)
        # # self.dense1 = nn.Linear(in_features=1024, out_features=128)
        # self.dense2 = nn.Linear(in_features=1024, out_features=tra_size)
        # self.bnd = nn.BatchNorm1d(tra_size)

        # small
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
        #                        kernel_size=5,
        #                        stride=1)
        # self.bn1 = nn.BatchNorm2d(4)
        # self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        # self.bn2 = nn.BatchNorm2d(8)
        # self.conv3 = nn.Conv2d(8, 16, kernel_size=5)
        # self.bn3 = nn.BatchNorm2d(16)
        # self.conv4 = nn.Conv2d(16, 32, kernel_size=5)
        # self.bn4 = nn.BatchNorm2d(32)
        # # self.dense1 = nn.Linear(in_features=1024, out_features=128)
        # self.dense2 = nn.Linear(in_features=512, out_features=tra_size)
        # self.bnd = nn.BatchNorm1d(tra_size)

        # smaller
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2,
                               kernel_size=(1, 5, 5),
                               stride=1)
        # self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=(1, 5, 5))
        # self.bn2 = nn.BatchNorm3d(4)
        self.conv3 = nn.Conv3d(4, 8, kernel_size=(1, 5, 5))
        # self.bn3 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(8, 16, kernel_size=(1, 5, 5))
        # self.bn4 = nn.BatchNorm2d(16)
        self.dense1 = nn.Linear(in_features=144, out_features=tra_size)
        self.dense2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=256, out_features=tra_size)
        self.dense3 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):
        # td
        # print x.size()
        t, n = x.size(0), x.size(1)
        # new_tuple = (t * n,) + x.size()[2:]
        x = x.permute(0, 2, 1, 3, 4)
        # print x.size()
        x = F.relu(F.max_pool3d(self.conv1(x), (1, 2, 2)))
        # print x.size()
        x = F.relu(F.max_pool3d(self.conv2(x), (1, 2, 2)))
        # print x.size()
        x = F.relu(F.max_pool3d(self.conv3(x), (1, 2, 2)))
        # print x.size()
        # x = F.relu(F.max_pool3d(self.bn4(self.conv4(x)), (1, 2, 2)))
        # print x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(t, n, 144)
        # # print x.size()
        x = F.selu(self.dense1(x))
        x = F.selu(self.dense2(x))
        x = F.selu(self.dense3(x))
        # # print "done dense 1"
        # x = F.selu(x)
        # # print "done selu 1"
        # x = self.dense2(x)
        # # print "done dense 2"
        # x = x.contiguous().permute(0, 2, 1).contiguous()
        # # print "done permute 1"
        # x = self.bnd(x)
        # # print "done bdn"
        # x = x.contiguous().permute(0, 2, 1).contiguous()
        # # print "done permute 2"
        # x = F.selu(x)
        # # print "done act"

        return x
        # ret = []
        # for t in range(x.size()[1]):
        #     _x = x[:, t]

        # WITH BN
        # _x = F.relu(F.max_pool2d(self.bn1(self.conv1(_x)), 2))
        # _x = F.relu(F.max_pool2d(self.bn2(self.conv2(_x)), 2))
        # _x = F.relu(F.max_pool2d(self.bn3(self.conv3(_x)), 2))
        # _x = F.relu(F.max_pool2d(self.bn4(self.conv4(_x)), 2))

        # NO BN
        # _x = F.relu(F.max_pool2d(self.conv1(_x), 2))
        # _x = F.relu(F.max_pool2d(self.conv2(_x), 2))
        # _x = F.relu(F.max_pool2d(self.conv3(_x), 2))
        # _x = F.relu(F.max_pool2d(self.conv4(_x), 2))
        # print _x.size()
        # _x = _x.view(-1, 1152)
        # _x = _x.view(-1, 512)
        # _x = _x.view(-1, 1024)
        # _x = self.dense1(_x)
        # _x = F.selu(_x)
        # _x = self.dense2(_x)
        # _x = self.bnd(_x)
        # _x = F.selu(_x)
        # ret.append(_x)

        # return torch.stack(ret, 1)


def out_shape(inp_size, kernels, max_pools):
    inp_size = list(inp_size)
    for k, m in zip(kernels, max_pools):
        inp_size[0] -= (k[0] - 1)
        inp_size[0] /= m[0] if m[0] is not None else 1
        # inp_size[0] = int(inp_size[0])
        inp_size[1] -= (k[1] - 1)
        inp_size[1] /= m[1] if m[1] is not None else 1
        # inp_size[1] = int(inp_size[0])

    return inp_size[0] * inp_size[1]


class cnn1_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(cnn1_transformation_module, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4,
                               kernel_size=(1, 5, 5),
                               stride=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(1, 5, 5))
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=(1, 5, 5))
        self.bn3 = nn.BatchNorm3d(16)

        self.out_feat = 16 * out_shape(inp_size[1:], [(5, 5)]*3, [(2, 2)]*3)

        self.dense1 = nn.Linear(in_features=self.out_feat, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.dense3 = nn.Linear(in_features=tra_size, out_features=tra_size)

        self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):

        t, n = x.size(0), x.size(1)
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(F.max_pool3d(self.bn1(self.conv1(x)), (1, 2, 2)))
        x = F.relu(F.max_pool3d(self.bn2(self.conv2(x)), (1, 2, 2)))
        x = F.relu(F.max_pool3d(self.bn3(self.conv3(x)), (1, 2, 2)))

        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(t, n, self.out_feat)

        x = self.dense1(x)
        x = x.contiguous().permute(0, 2, 1).contiguous()
        x = self.bnd(x)
        x = x.contiguous().permute(0, 2, 1).contiguous()
        x = F.selu(x)

        return x


class cnn2_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(cnn2_transformation_module, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2,
                               kernel_size=(1, 5, 5),
                               stride=1)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=(1, 5, 5))
        self.bn2 = nn.BatchNorm3d(4)
        self.conv3 = nn.Conv3d(4, 8, kernel_size=(1, 5, 5))
        self.bn3 = nn.BatchNorm3d(8)
        self.out_feat = 8 * out_shape(inp_size[1:], [(5, 5)]*3, [(2, 2)]*3)
        self.dense1 = nn.Linear(in_features=self.out_feat, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=256, out_features=tra_size)
        # self.dense3 = nn.Linear(in_features=tra_size, out_features=tra_size)
        self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(F.max_pool3d(self.bn1(self.conv1(x)), (1, 2, 2)))
        x = F.relu(F.max_pool3d(self.bn2(self.conv2(x)), (1, 2, 2)))
        x = F.relu(F.max_pool3d(self.bn3(self.conv3(x)), (1, 2, 2)))
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(t, n, self.out_feat)
        x = self.dense1(x)
        x = x.contiguous().permute(0, 2, 1).contiguous()
        x = self.bnd(x)
        x = x.contiguous().permute(0, 2, 1).contiguous()

        x = F.selu(x)

        return x


class cnn3_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(cnn3_transformation_module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=(5, 5),
                               stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(5, 5))
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.bn4 = nn.BatchNorm2d(32)
        self.dense1 = nn.Linear(in_features=64, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=256, out_features=tra_size)
        # self.dense3 = nn.Linear(in_features=tra_size, out_features=tra_size)
        self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):

        ret = []
        for t in range(x.size()[1]):
            _x = x[:, t]
            _x = F.relu(F.max_pool2d(self.bn1(self.conv1(_x)), 2))
            _x = F.relu(F.max_pool2d(self.bn2(self.conv2(_x)), 2))
            _x = F.relu(F.max_pool2d(self.bn3(self.conv3(_x)), 2))
            _x = _x.view(-1, 64)
            _x = self.dense1(_x)
            _x = self.bnd(_x)
            _x = F.selu(_x)
            ret.append(_x)

        return torch.stack(ret, 1)

class cnn4_transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(cnn4_transformation_module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2,
                               kernel_size=(5, 5),
                               stride=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(5, 5))
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(5, 5))
        self.bn4 = nn.BatchNorm2d(16)
        self.dense1 = nn.Linear(in_features=144, out_features=tra_size)
        self.dense2 = nn.Linear(in_features=tra_size, out_features=tra_size)
        # self.dense2 = nn.Linear(in_features=256, out_features=tra_size)
        self.dense3 = nn.Linear(in_features=tra_size, out_features=tra_size)
        self.bnd = nn.BatchNorm1d(tra_size)

    def forward(self, x):

        ret = []
        for t in range(x.size()[1]):
            _x = x[:, t]
            _x = F.relu(F.max_pool2d(self.bn1(self.conv1(_x)), 2))
            _x = F.relu(F.max_pool2d(self.bn2(self.conv2(_x)), 2))
            _x = F.relu(F.max_pool2d(self.bn3(self.conv3(_x)), 2))
            _x = _x.view(-1, 144)
            _x = self.dense1(_x)
            _x = self.bnd(_x)
            _x = F.selu(_x)
            ret.append(_x)

        return torch.stack(ret, 1)


class identity_module(nn.Module):
    def __init__(self):
        super(identity_module, self).__init__()

    def forward(self, x):
        return x


class attention_module(nn.Module):
    def __init__(self, inp_size, att_size, rnn_mode='LSTM'):
        super(attention_module, self).__init__()

        # Transformation
        if rnn_mode == 'LSTM':
            self.rnn = nn.LSTM(input_size=inp_size, hidden_size=att_size, num_layers=1, bias=True,
                               bidirectional=False)
        if rnn_mode == 'GRU':
            self.rnn = nn.GRU(input_size=inp_size, hidden_size=att_size, num_layers=1, bias=True,
                              bidirectional=False)
        # elif rnn_mode == 'SRU':
        #     self.rnn = SRU(input_size=inp_size, hidden_size=att_size, num_layers=1,
        #                       bidirectional=False)

        self.fc = nn.Linear(in_features=att_size, out_features=1)

    def forward(self, x, x_len):
        x = x.transpose(0, 1)
        out, _ = self.rnn(x)
        out = out.transpose(0, 1)
        out = self.fc(out)
        out = F.selu(out)

        return out


class merge_module(nn.Module):
    def __init__(self, num_sensors, disabled=[]):
        super(merge_module, self).__init__()
        self.num_sensors = num_sensors
        self.disabled = disabled
        self.mask = [0.0 if element in self.disabled else 1.0 for element in range(self.num_sensors)]

    def forward(self, x_attention, x_transform):
        # MINOR MESS WARNING: pytorch does not support dim argument in softmax, hence the transposing.
        softmax_attention = F.softmax(torch.cat(x_attention, -1).transpose(0, 2)).transpose(0, 2)
        split_attention = softmax_attention.split(1, 2)

        scale = []
        for idx in range(self.num_sensors):
            if idx not in self.disabled:
                scale.append(split_attention[idx] * x_transform[idx])
            else:
                scale.append(0 * x_transform[idx])

        merge_stack = torch.stack(scale)
        merge = merge_stack.sum(0)

        return merge, list(split_attention), scale

class classification_module(nn.Module):
    def __init__(self, inp_size, cla_size, num_classes, num_layers, rnn_mode='LSTM', cla_dropout=0.0):
        super(classification_module, self).__init__()
        if rnn_mode == 'LSTM':
            self.rnn = nn.LSTM(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers, bias=True,
                               bidirectional=True, dropout=cla_dropout)
        if rnn_mode == 'GRU':
            self.rnn = nn.GRU(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers, bias=True,
                              bidirectional=True, dropout=cla_dropout)
        elif rnn_mode == 'SRU':
            self.rnn = SRU(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers,
                           bidirectional=True, dropout=cla_dropout, rnn_dropout=cla_dropout)
        self.fc = nn.Linear(in_features=2 * cla_size, out_features=num_classes, bias=True)

    def forward(self, x, x_len):
        x = x.transpose(0, 1)
        out_rnn, _ = self.rnn(x)
        out_rnn = out_rnn.transpose(0, 1)
        out_fc = self.fc(out_rnn)

        return out_rnn, out_fc

class classification_module_cla(nn.Module):
    def __init__(self, inp_size, cla_size, num_classes, num_layers, rnn_mode='LSTM', cla_dropout=0.0, bidirectional=True):
        super(classification_module_cla, self).__init__()
        if rnn_mode == 'LSTM':
            self.rnn = nn.LSTM(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers, bias=True,
                               bidirectional=bidirectional, dropout=cla_dropout)
        if rnn_mode == 'GRU':
            self.rnn = nn.GRU(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers, bias=True,
                              bidirectional=bidirectional, dropout=cla_dropout)
        # elif rnn_mode == 'SRU':
        #     self.rnn = SRU(input_size=inp_size, hidden_size=cla_size, num_layers=num_layers,
        #                        bidirectional=True, dropout=cla_dropout,rnn_dropout=cla_dropout)
        self.fc = nn.Linear(in_features=2 * cla_size if bidirectional else cla_size, out_features=num_classes, bias=True)

    def forward(self, x, x_len):
        x = x.transpose(0, 1)
        # print x.size()
        out_rnn, _ = self.rnn(x)
        # print out_rnn.size()
        out_rnn = out_rnn[-1]
        # print out_rnn.size()
        # out_rnn = out_rnn.transpose(0,1)
        out_fc = self.fc(out_rnn)
        # print out_fc.size()
        # print out_fc.size()
        return out_rnn, out_fc


class audio_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type='dense', rnn_mode='LSTM', cla_dropout=0.0):
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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        if tra_type == 'dense':
            self.transformation_module = nn.ModuleList()
            for idx in range(self.num_sensors):
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

        if tra_type == 'rnn':
            self.transformation_module = nn.ModuleList()
            for idx in range(self.num_sensors):
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

        if tra_type == 'identity':
            self.transformation_module = nn.ModuleList()
            for idx in range(self.num_sensors):
                self.transformation_module.append(identity_module())
            self.tra_size = self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            self.attention_module.append(
                attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        attention = []
        for trans, a_module in zip(transformation, self.attention_module):
            attention.append(a_module(trans, x_len))

        # Merge
        merge, sm_attention, scale = self.merge_module(attention, transformation)

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        # ctc_out = dense.transpose(0, 1)
        out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': out,
                 'transforms': transformation, 'attention': attention,
                 'sm_attentions': sm_attention,
                 'scale': scale, 'merge': merge}

        return out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy


# class audio_single(nn.Module):
#     def __init__(self, num_sensors, inp_size, tra_size=50, cla_size=150, cla_layers=2, num_classes=59,
#                  tra_type='dense', rnn_mode='LSTM', cla_dropout=0.0):
#         super(audio_single, self).__init__()
#
#         # Sensor parameters
#         self.num_sensors = num_sensors
#         self.inp_size = inp_size[0]
#
#         # Transformation module parameters
#         self.tra_size = tra_size
#         self.tra_type = tra_type
#
#         # Classification module parameters
#         self.cla_size = cla_size
#         self.cla_layers = cla_layers
#         self.classes = num_classes
#         self.cla_dropout = cla_dropout
#
#         # Overall parameters
#         self.rnn_mode = rnn_mode
#
#         # List of transformation modules
#         if tra_type == 'dense':
#             self.transformation_module=transformation_module(inp_size=inp_size[0], tra_size=self.tra_size)
#
#         if tra_type == 'rnn':
#             self.transformation_module=rnn_transformation_module(inp_size=inp_size[0], tra_size=self.tra_size)
#
#         if tra_type == 'identity':
#             self.transformation_module=identity_module()
#
#         # Classification module
#         self.classification_module = classification_module(inp_size=self.inp_size, cla_size=self.cla_size,
#                                                            num_layers=self.cla_layers, num_classes=self.classes,
#                                                            rnn_mode=self.rnn_mode, cla_dropout = self.cla_dropout)
#
#     def forward(self, x, x_len):
#
#         # Transformation
#         transformation = [self.transformation_module(x[0])] # list
#
#         # Classification
#         classifier, dense = self.classification_module(transformation[0], x_len) # no list
#
#         ctc_out = dense.transpose(0, 1) # no list
#
#         ############ Placeholders
#         attention = [transformation[0][:, :, :1]] # list
#         merge = transformation[0] # no list
#         scale = [transformation[0]] # list
#         sm_attention = attention # no list
#
#         # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
#         debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
#                       'transforms': transformation, 'attention': attention,
#                       'sm_attentions': sm_attention,
#                       'scale': scale, 'merge': merge}
#
#         return ctc_out, debug
#
#     def to_numpy(self, object):
#         if torch.is_tensor(object) == False:
#             object = object.data
#         if object.is_cuda:
#             object = object.cpu()
#         return object.numpy()
#
#     def debug_to_numpy(self, debug):
#         debug_numpy = {}
#         for key, value in debug.iteritems():
#             if type(value) == list:
#                 numpy_list = []
#                 for element in value:
#                     numpy_list.append(self.to_numpy(element))
#                 debug_numpy[key] = numpy_list
#             else:
#                 debug_numpy[key] = self.to_numpy(value)
#         return debug_numpy


class audio_video_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('dense', 'cnn'), rnn_mode='LSTM', cla_dropout=0.0):
        super(audio_video_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn':
                self.transformation_module.append(
                    cnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            self.attention_module.append(
                attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        attention = []
        for trans, a_module in zip(transformation, self.attention_module):
            attention.append(a_module(trans, x_len))

        # Merge
        merge, sm_attention, scale = self.merge_module(attention, transformation)

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        ctc_out = dense.transpose(0, 1)
        # out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
                 'transforms': transformation, 'attention': attention,
                 'sm_attentions': sm_attention,
                 'scale': scale, 'merge': merge}

        return ctc_out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy


class video_video_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('cnn1', 'cnn2'), rnn_mode='LSTM', cla_dropout=0.0):
        super(video_video_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn1':
                self.transformation_module.append(
                    cnn1_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn2':
                self.transformation_module.append(
                    cnn2_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn3':
                self.transformation_module.append(
                    cnn3_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn4':
                self.transformation_module.append(
                    cnn4_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        at = attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode)
        for idx in range(self.num_sensors):
            self.attention_module.append(
                at)

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        attention = []
        for trans, a_module in zip(transformation, self.attention_module):
            attention.append(a_module(trans, x_len))

        # Merge
        merge, sm_attention, scale = self.merge_module(attention, transformation)

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        ctc_out = dense.transpose(0, 1)
        # out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
                 'transforms': transformation, 'attention': attention,
                 'sm_attentions': sm_attention,
                 'scale': scale, 'merge': merge}

        return ctc_out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy


class single_video_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('cnn1'), rnn_mode='LSTM', cla_dropout=0.0):
        super(single_video_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn1':
                self.transformation_module.append(
                    cnn1_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn2':
                self.transformation_module.append(
                    cnn2_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn3':
                self.transformation_module.append(
                    cnn3_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn4':
                self.transformation_module.append(
                    cnn4_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # List of attention modules
        # self.attention_module = nn.ModuleList()
        # for idx in range(self.num_sensors):
        #     self.attention_module.append(
        #         attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        # attention = []
        # for trans, a_module in zip(transformation, self.attention_module):
        #     attention.append(a_module(trans, x_len))
        #
        # # Merge
        # merge, sm_attention, scale = self.merge_module(attention, transformation)

        merge = transformation[0]

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        ctc_out = dense.transpose(0, 1)
        # out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
                 'transforms': transformation, 'merge': merge}

        return ctc_out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy



class video_video_cla_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('cnn1', 'cnn2'), rnn_mode='LSTM', cla_dropout=0.0):
        super(video_video_cla_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn1':
                self.transformation_module.append(
                    cnn1_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn2':
                self.transformation_module.append(
                    cnn2_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn3':
                self.transformation_module.append(
                    cnn3_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn4':
                self.transformation_module.append(
                    cnn4_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        at = attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode)
        for idx in range(self.num_sensors):
            self.attention_module.append(at)

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module_cla(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout, bidirectional=False)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        attention = []
        for trans, a_module in zip(transformation, self.attention_module):
            attention.append(a_module(trans, x_len))

        # Merge
        merge, sm_attention, scale = self.merge_module(attention, transformation)

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        # ctc_out = dense.transpose(0, 1)
        out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': out,
                 'transforms': transformation, 'attention': attention,
                 'sm_attentions': sm_attention,
                 'scale': scale, 'merge': merge}

        return out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy



class single_video_cla_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('cnn1', 'cnn2'), rnn_mode='LSTM', cla_dropout=0.0):
        super(single_video_cla_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn1':
                self.transformation_module.append(
                    cnn1_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn2':
                self.transformation_module.append(
                    cnn2_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn3':
                self.transformation_module.append(
                    cnn3_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn4':
                self.transformation_module.append(
                    cnn4_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # # List of attention modules
        # self.attention_module = nn.ModuleList()
        # for idx in range(self.num_sensors):
        #     self.attention_module.append(
        #         attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        # Classification module
        self.classification_module = classification_module_cla(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout, bidirectional=False)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # # Attention
        # attention = []
        # for trans, a_module in zip(transformation, self.attention_module):
        #     attention.append(a_module(trans, x_len))
        #
        # # Merge
        # merge, sm_attention, scale = self.merge_module(attention, transformation)
        merge = transformation[0]

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        # ctc_out = dense.transpose(0, 1)
        out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': out,
                 'transforms': transformation, 'merge': merge}

        return out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy

class single_audio_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, cla_size=150, cla_layers=2, num_classes=59,
                 tra_type=('dense',), rnn_mode='LSTM', cla_dropout=0.0):
        super(single_audio_stan, self).__init__()

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
        self.cla_dropout = cla_dropout

        # Overall parameters
        self.rnn_mode = rnn_mode

        # List of transformation modules
        self.transformation_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            if tra_type[idx] == 'dense':
                self.transformation_module.append(transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'rnn':
                self.transformation_module.append(
                    rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn':
                self.transformation_module.append(
                    cnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size = self.inp_size[0]

        # # List of attention modules
        # self.attention_module = nn.ModuleList()
        # for idx in range(self.num_sensors):
        #     self.attention_module.append(
        #         attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))


        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # Attention
        # attention = []
        # for trans, a_module in zip(transformation, self.attention_module):
        #     attention.append(a_module(trans, x_len))

        # Merge
        # merge, sm_attention, scale = self.merge_module(attention, transformation)
        merge = transformation[0]

        # Classification
        classifier, dense = self.classification_module(merge, x_len)

        ctc_out = dense.transpose(0, 1)
        # out = dense

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
                 'transforms': transformation}

        return ctc_out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.iteritems():
            if type(value) == list:
                numpy_list = []
                for element in value:
                    numpy_list.append(self.to_numpy(element))
                debug_numpy[key] = numpy_list
            else:
                debug_numpy[key] = self.to_numpy(value)
        return debug_numpy
