from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import pandas as pd
from torch.autograd import Variable


# from the gcn paper
def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I
    adjacencies = sp.csr_matrix(adjacencies, dtype=np.float32)
    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]
    return adj_norm

    # Reconstruction + KL divergence losses summed over all elements and batch
    
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph_L(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_normalized = sp.csc_matrix.todense(adj_normalized)
    return adj_normalized

#loss function  (predict, label)
def loss_function(pre_adj, adj):
    adj = torch.Tensor(adj)
    adj_shape = adj.size()
    # init_weight = nn.init.kaiming_uniform_(torch.Tensor(adj_shape[0], adj_shape[1]))
    class_weight = Variable(torch.FloatTensor([1, 200])) # the proportion of positive and negative samples, which is a manual adjustment parameters
    weight = class_weight[adj.long()]
    # weight = weight.mul(torch.abs(init_weight))
    # loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    loss_fn = torch.nn.BCELoss(weight)
    return loss_fn(pre_adj, adj)

# no use
def dis_loss(pred, label):
    fn = torch.nn.BCELoss()
    loss = fn(pred, label)
    return loss

# train method
def train(epoch, label, encoder, decoder, discriminator, device, optimizer, lncx, disx, adj, dis_optimizer, row, col):

    # train
    optimizer.zero_grad()
    z = encoder(lncx, disx, adj)
    row_n = len(row)
    col_n = len(col)
    feature = torch.split(z, [row_n, col_n], dim=0)
    lnc = feature[0]
    dis = feature[1]

    out = decoder(lnc, dis)
    pred = out

    en_loss = loss_function(pred, label)
    en_loss.backward()
    optimizer.step()

    return pred, en_loss


def gaussiansim (a,b, sigma=1):
    t = (a-b)**2
    temp = -sum(t)

    return np.exp(temp/ (2 * sigma**2)) # the denominator view as a constant, which is equivalent to the sum function in our paper 
