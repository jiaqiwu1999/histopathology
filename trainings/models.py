import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as torch_models
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

# Custom Model
class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, (5,5), (2,2))
        self.conv2 = nn.Conv2d(8, 32, (3,3))
        self.conv3 = nn.Conv2d(16, 64, (3,3))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.predict = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(self.predict(x))
        return x



# from GeneMutationFromHE
class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        return x
    
    
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( 1 X C X N)
            returns :
                out : self attention value + input feature
                attention:  1 X N X N  (N is number of patches)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)  # 1 X N X C
        proj_key = self.key_conv(x)  # 1 X C x N
        energy = torch.bmm(proj_query, proj_key)  # 1 X N X N
        attention = self.softmax(energy)  # 1 X N X N

        out = torch.bmm(x, attention.permute(0, 2, 1))
        out = self.gamma * out + x

        return out, attention


class AttnClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AttnClassifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.attn = Self_Attn(128)
        self.fc3 = nn.Linear(128, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, attn = self.attn(x.permute(1, 2, 0))
        x2 = x.mean(dim=-1)
        x = self.fc3(x2)

        return x, attn

# ------------- Simplified Inception v3 --------------
class Inception_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = torch_models.inception_v3()
    
    def forward(self, x):
         # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x
    
    
    
    
#---------------Over complicated Inception model ..-------------------------

class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    
    
class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0,
                 activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
    
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels=1, name='Inception'):
        super(InceptionModule, self).__init__()
        
        self.b0 = Unit3D(in_channels, out_channels[0], [1,1,1], 0, name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels, out_channels[1], [1,1,1], 0, name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(out_channels[1], out_channels[2], [3,3,3], 0, name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels, out_channels[3], [1,1,1], 0, name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(out_channels[3], out_channels[4], [3,3,3], 0, name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding([3,3,3], [1,1,1], 0)
        self.b3b = Unit3D(in_channels, out_channels[5], [1,1,1], 0, name+'/Branch_3/Conv3d_0b_1x1')
                          
                          
    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)
                          
                                    
                        
class Inception3D(nn.Module):
    def __init__(self, num_classes=1, name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        super(Inception3D, self).__init__()
        self.endpoints = {}
        self.endpoints['Conv3d_1a_7x7'] = Unit3D(in_channels, 64, [7,7,7], (2,2,2), (3,3,3), name+'Conv3d_1a_7x7')
        self.endpoints['MaxPool3d_2a_3x3'] = MaxPool3dSamePadding([1,3,3], (1,2,2), 0)
        self.endpoints['Conv3d_2b_1x1'] = Unit3D(64, 64, [1,1,1], 0, name+'Conv3d_2b_1x1')
        self.endpoints['Conv3d_2c_3x3'] = Unit3D(64, 192, [3,3,3], 1, name+'Conv3d_2c_3x3')
        self.endpoints['MaxPool3d_3a_3x3'] = MaxPool3dSamePadding([1,3,3], (1,2,2), 0)
        self.endpoints['Mixed_3b'] = InceptionModule(192, [64,96,128,16,32,32], name+'Mixed_3b')
        self.endpoints['Mixed_3c'] = InceptionModule(256, [128,128,192,32,96,64], name+'Mixed_3c')
        self.endpoints['MaxPool3d_4a_3x3'] = MaxPool3dSamePadding([3,3,3], (2,2,2), 0)
        self.endpoints['Mixed_4b'] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+'Mixed_4b')
        self.endpoints['Mixed_4c'] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+'Mixed_4c')
        self.endpoints['Mixed_4d'] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+'Mixed_4d')
        self.endpoints['Mixed_4e'] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+'Mixed_4e')
        self.endpoints['Mixed_4f'] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+'Mixed_4f')
        self.endpoints['MaxPool3d_5a_2x2'] = MaxPool3dSamePadding([2, 2, 2], (2, 2, 2), 0)
        self.endpoints['Mixed_5b'] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+'Mixed_5b')
        self.endpoints['Mixed_5c'] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+'Mixed_5c')
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=1,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
                          
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
            
    def forward(self, x):
        pass
        
# ---------------------------------------------
        
class Simple_3D_model(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(simple_3D_model, self).__init__()
        # input shape: N, 3, 224, 224
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(7,7,7), stride=(2,2,2), padding=(3,3,3))#64, 32, 32
        self.conv2 = nn.Conv3d(64, 128, (3,3,3), (1,1,1), padding='same') # 128, 32, 32
        self.pool1 = nn.MaxPool3d((1,3,3), (1,2,2)) # 128, 16, 16
        self.conv3 = nn.Conv3d(128, 64, (1,1,1), (1,1,1), padding='same') # 64, 16, 16
        self.conv4 = nn.Conv3d(64, 32, (1,1,1), (1,1,1), padding='same') # 32, 16, 16
        self.pool2 = nn.MaxPool3d((1,3,3), (1,2,2)) # 32, 8, 8
        self.glob = nn.AvgPool2d((8,8))
        self.dense1 = nn.Dense(32, 16)
        self.dense2 = nn.Dense(16, num_classes)
    
    def forward(self, x):
        N, C, H, W = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.glob(x)
        print(x.shape) # N, 32
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    

class AttnClassifier(nn.Module):
    def __init__(self, num_class=1, input_dim=2048, hidden_dim=128):
        super(AttnClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, hidden_dim)
        self.alpha = nn.Linear(hidden_dim, 1)
        self.pred = nn.Linear(hidden_dim, num_class)
        self.activation = F.sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # b x 32 x 2048
        x1 = self.fc1(x) # b x 32 x 512
        x2 = self.fc2(x1) # b x 32 x 128
        att = self.alpha(x2) # b x 32 x 1
        normed_att = self.softmax(att) # b x 32 x 1
        combined = torch.bmm(x2.permute(0, 2, 1), normed_att).squeeze(2) # (b, 128, 32) x (b, 32, 1) -> b, 128, 1 -> b, 128
        fin = self.activation(self.pred(combined))
        return fin

# ------------- RNN -------------------
class RNNAttUnit(nn.Module):
    def __init__(self, input_dim, output_dim, size=(200,200), bias=False):
        super(RNNAttUnit, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameters((input_dim, output_dim))
        self.att = Parameters(size)
        self.activation = F.Tanh() # or Tanh?
        self.softmax = nn.Softmax(dim=1)
        if self.bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        support = torch.matmul(x, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class RNNAttBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, size, dropout):
        super(RNNAttBlock, self).__init__()
        self.unit1 = RNNAttUnit(input_dim, input_dim, size)
        self.in_features = input_dim
        self.out_features = input_dim
        self.is_resi = is_resi

        self.gc1 = RNNAttUnit(in_features, in_features, size)
        self.bn1 = nn.BatchNorm1d(size[0] * in_features)

        self.gc2 = RNNAttUnit(in_features, in_features, size)
        self.bn2 = nn.BatchNorm1d(size[0] * in_features)

        self.do = nn.Dropout(dropout)
        self.act_f = nn.Tanh()
        
    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y + x

        

class RNNModel(nn.Module):
    def __init__(self, num_layers, layer_type, sequence_length, num_class=1, input_dim=2048, hidden_dim=128, dropout=0.2):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        self.activation = F.sigmoid()
        self.fc = nn.Linear(hidden_dim, num_class)
        self.rnn_list = nn.ModuleList()
        for i in range(num_layers):
            if cell_type == 'GRU':
                rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            elif cell_type == 'LSTM':
                rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            else:
                print("Cell type not supported")
                return
            self.rnn_list.append(rnn)
        self.h, self.c = self.initialize()
        self.attention = RNNAttBlock() #?
        
    def initialize():
        h0 = torch.empty((self.num_layers, self.sequence_length, self.hidden_dim))
        c0 = h0
        return torch.nn.init.normal_(h0), torch.nn.init.normal_(c0)
        
        
    def forward(self, x):
        # shape of x: batch x length x input_dim
        h = self.h
        c = self.c
        layer_final_hidden_state = []
        layer_final_cell_state = []
        for rnn in self.rnn_list:
            output, (h, c) = rnn(x, (h, c))
            layer_final_hidden_state.append(h)
            layer_final_cell_state.append(c)
        layer_final_hidden_state = torch.concat(layer_final_hidden_state, dim=0) # num_layer x batch x hidden_dim
        # what to do with this? attention?
        out = self.attention(layer_final_hidden_state.permute(1, 0, 2))
        #?
        return out
        
        
        
        
    