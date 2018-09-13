import torch
import torch.nn as nn
import torch.nn.functional as F


# from cuda_functional import SRU

class transformation_module(nn.Module):
    def __init__(self, inp_size, tra_size):
        super(transformation_module, self).__init__()
        self.fc = nn.Linear(in_features=inp_size, out_features=tra_size)

    def forward(self, x):
        out = self.fc(x)
        out = F.selu(out)

        return out


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

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1620, tra_size)


    def forward(self, x):
        t, n = x.size(0), x.size(1)
        new_tuple = (t * n,) + x.size()[2:]

        x = x.contiguous().view(new_tuple)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1620)
        x = F.selu(self.fc1(x))
        new_tuple = (t, n) + x.size()[1:]
        x = x.contiguous().view(new_tuple)
        return x



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
        elif rnn_mode == 'SRU':
            self.rnn = SRU(input_size=inp_size, hidden_size=att_size, num_layers=1,
                           bidirectional=False)

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

    def forward(self, x_attention, x_transform):
        # Disable sensors
        if self.disabled:
            remainder = set(range(self.num_sensors)) ^ set(self.disabled)
            for idx in sorted(self.disabled, reverse=True):
                del x_attention[idx], x_transform[idx]
        num_active = len(x_attention)

        # Softmax attention weights
        # MINOR MESS WARNING: pytorch does not support dim argument in softmax, hence the transposing.
        softmax_attention = F.softmax(torch.cat(x_attention, -1).transpose(0, 2)).transpose(0, 2)
        split_attention = softmax_attention.split(1, 2)

        # If one sensor only: convert weird pytorch tuple to list
        if num_active == 1:
            split_attention = [split_attention[0]]

        # Scale transforms by attention weights
        scale = []
        for idx in range(num_active):
            scale.append(split_attention[idx] * x_transform[idx])

        # Weighted sum
        merge_stack = torch.stack(scale)
        merge = merge_stack.sum(0)

        # STD hack
        # import matplotlib.pyplot as plt
        # plt.imshow(merge.cpu().data.numpy()[0,:,:].T)
        # plt.colorbar()
        # plt.show()
        # print('Before {} {}'.format(torch.mean(merge).cpu().data.numpy(),torch.std(merge).cpu().data.numpy()))
        # merge = 0.85/torch.std(merge)*merge
        # print('After {}'.format(torch.std(merge).cpu().data.numpy()))

        # Fill attention and scale lists with disabled sensors. Sorry, really messy
        if self.disabled:
            temp_split_attention = [split_attention[0] * 0] * self.num_sensors
            temp_scale = [scale[0] * 0] * self.num_sensors
            for sca, att, idx in zip(scale, split_attention, remainder):
                temp_scale[idx] = sca
                temp_split_attention[idx] = att
            split_attention = temp_split_attention
            scale = temp_scale

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


class audio_stan(nn.Module):
    def __init__(self, num_sensors, inp_size, tra_size=50, att_size=20, att_share=False, cla_size=150, cla_layers=2,
                 num_classes=59,
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
        self.att_share = att_share

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
        # self.attention_module = nn.ModuleList()
        # if self.att_share == True:
        #     print('{}\nSHARED-MODE: ATTENTION MODULES USE -SHARED- WEIGHTS\n{}'.format('!'*50,'!'*50))
        #     shared_module = attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode)
        #     for idx in range(self.num_sensors):
        #         self.attention_module.append(shared_module)
        # else:
        #     for idx in range(self.num_sensors):
        #         self.attention_module.append(
        #             attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        # self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout=self.cla_dropout)

    def forward(self, x, x_len):

        # Transformation
        transformation = []
        for sensor, t_module in zip(x, self.transformation_module):
            transformation.append(t_module(sensor))

        # # Attention
        # attention = []
        # for trans, a_module in zip(transformation, self.attention_module):
        #     # print(torch.sum(a_module.rnn.all_weights[0][0]))
        #     attention.append(a_module(trans, x_len))
        #
        # # Merge
        # merge, sm_attention, scale = self.merge_module(attention, transformation)


        # Classification
        classifier, dense = self.classification_module(transformation[0], x_len)

        ctc_out = dense.transpose(0, 1)

        # Probably not required as stan is object oriented and pytorch is imperative! Amazing.
        debug = {'inputs': x, 'classifier': classifier, 'dense': dense, 'output': ctc_out,
                 'transforms': transformation, } #'attention': attention,
                 # 'sm_attentions': sm_attention,
                 # 'scale': scale, 'merge': merge}

        return ctc_out, debug

    def to_numpy(self, object):
        if torch.is_tensor(object) == False:
            object = object.data
        if object.is_cuda:
            object = object.cpu()
        return object.numpy()

    def debug_to_numpy(self, debug):
        debug_numpy = {}
        for key, value in debug.items():
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
                 tra_type=('dense', 'dense'), rnn_mode='LSTM', cla_dropout=0.0):
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
                self.transformation_module.append(rnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'cnn':
                self.transformation_module.append(cnn_transformation_module(inp_size=inp_size[idx], tra_size=self.tra_size))

            if tra_type[idx] == 'identity':
                self.transformation_module.append(identity_module())
                self.tra_size=self.inp_size[0]

        # List of attention modules
        self.attention_module = nn.ModuleList()
        for idx in range(self.num_sensors):
            self.attention_module.append(attention_module(inp_size=self.tra_size, att_size=self.att_size, rnn_mode=self.rnn_mode))

        # Merge module
        self.merge_module = merge_module(num_sensors=self.num_sensors)

        # Classification module
        self.classification_module = classification_module(inp_size=self.tra_size, cla_size=self.cla_size,
                                                           num_layers=self.cla_layers, num_classes=self.classes,
                                                           rnn_mode=self.rnn_mode, cla_dropout = self.cla_dropout)

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

