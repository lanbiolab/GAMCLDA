import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
#from torch_geometric.nn.inits import uniform

import methods
import numpy as np

import math


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    # stdv = math.sqrt(6.0 / size)
    if tensor is not None:
        # tensor.data.uniform_(-stdv, stdv)#nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(tensor)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, bias=False):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        #self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def init_weight(self):
    #
    #     for param in self.parameters():
    #         param.data.normal_(1 / param.size(1) ** 0.5)
    #         param.data.renorm_(2, 0, 1)
    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        # uniform(self.lncrna_in_channels, self.lnc_bias)
        # uniform(self.disease_in_channels, self.dis_bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        adj_norm = methods.preprocess_graph_L(adj) #compute A hat
        temp_x = torch.matmul(x, self.weight)
        list1 = []
        # for a in adj_norm:
        #     list1.append(a.toarray())
        #adj_norm = torch.Tensor(np.array(list1).reshape(652,652))
        adj_norm = torch.Tensor(adj_norm)
        output = torch.matmul(adj_norm, temp_x)

        if self.bias is not None:
            output = output + self.bias

        return output


class Dense(torch.nn.Module):

    def __init__(self, in_channels, out_channels, normalize=False, bias=False):
        super(Dense, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lnc_weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.dis_weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        #self.bias = bias
        if bias:
            self.lnc_bias = Parameter(torch.Tensor(out_channels))
            self.dis_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def init_weight(self):
    #
    #     for param in self.parameters():
    #         param.data.normal_(1 / param.size(1) ** 0.5)
    #         param.data.renorm_(2, 0, 1)
    def reset_parameters(self):
        uniform(self.in_channels, self.lnc_weight)
        uniform(self.in_channels, self.dis_weight)
        # uniform(self.in_channels, self.bias)

    def forward(self, lnc_x, dis_x):
        lnc_output = torch.matmul(lnc_x, self.lnc_weight)
        dis_output = torch.matmul(dis_x, self.dis_weight)

        if self.bias is not None:
            lnc_output = lnc_output + self.lnc_bias
            dis_output = dis_output + self.dis_bias

        return lnc_output, dis_output


class FeatureDense(torch.nn.Module):

    def __init__(self, in_channels, out_channels, normalize=False, bias=False):
        super(FeatureDense, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        #self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def init_weight(self):
    #
    #     for param in self.parameters():
    #         param.data.normal_(1 / param.size(1) ** 0.5)
    #         param.data.renorm_(2, 0, 1)
    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        # uniform(self.in_channels, self.bias)

    def forward(self, x):
        output = torch.matmul(x, self.weight)

        if self.bias is not None:
            output = output + self.bias

        return output




