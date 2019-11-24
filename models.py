import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.nn as nn
#from torch_geometric.nn.inits import uniform
from layers import GCN
from layers import Dense
from layers import FeatureDense
import numpy as np

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    # stdv = math.sqrt(6.0 / size)
    if tensor is not None:
        # tensor.data.uniform_(-stdv, stdv)#nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(tensor)


class Encoder(torch.nn.Module):
    def __init__(self, lncrna_in_channels, disease_in_channels, hidden_1, hidden_2, hidden_3):
        super(Encoder, self).__init__()
        self.lncdense = FeatureDense(lncrna_in_channels, hidden_1)
        self.disdense = FeatureDense(disease_in_channels, hidden_1)

        self.lncdense2 = FeatureDense(500, hidden_1)
        self.disdense2 = FeatureDense(500, hidden_1)

        self.gcn1 = GCN(hidden_1, hidden_2)
        self.gcn2 = GCN(hidden_2, hidden_3)
        self.gcn3 = GCN(hidden_3, hidden_3)
        self.dense = FeatureDense(hidden_3, hidden_3)

    def forward(self, lncrna_x, disease_x, adj):
        # lncrna_x = F.dropout(lncrna_x)
        # disease_x = F.dropout(disease_x)
        lncx = self.lncdense(lncrna_x)
        disx = self.disdense(disease_x)

        lncx = F.relu(lncx)
        disx = F.relu(disx)
        x = torch.cat((lncx, disx))

        z = self.gcn1(x, adj)
        z = F.relu(z)
        z = self.gcn2(z, adj)
        return z


class Decoder(torch.nn.Module):

    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, lncx, disx):

        output = torch.matmul(lncx, torch.t(disx))

        # temp_o = torch.matmul(lncx, self.weight)
        # output = torch.matmul(temp_o, torch.t(disx))

        return F.sigmoid(output)






