from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sortscore
import h5py
import time
import methods
from models import Encoder
from models import Decoder
import itertools


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
with h5py.File('lncrna_disease_association.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]
    lncrna_disease_matrix_val = lncrna_disease_matrix.copy()
index_tuple = (np.where(lncrna_disease_matrix == 1))
one_list = list(zip(index_tuple[0], index_tuple[1]))
random.shuffle(one_list)
split = math.ceil(len(one_list) / 10)
all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []

hidden1 = 256 #32
hidden2 = 64
hidden3 = 32 #16

with h5py.File('lncRNA_feature.h5', 'r') as hf:
    lncx = hf['infor'][:]
    lncx = torch.Tensor(lncx)
with h5py.File('disease_feature.h5', 'r') as hf:
    disx = hf['infor'][:]
    disx = torch.Tensor(disx)
#10-fold start
for i in range(0, len(one_list), split):

    encoder = Encoder(lncx.shape[1], disx.shape[1], hidden1, hidden2, hidden3)
    decoder = Decoder(hidden3)
    gen_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.001, weight_decay=0.005)#0.01
    train_index = one_list[i:i + split]
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()

    for index in train_index:
        new_lncrna_disease_matrix[index[0], index[1]] = 0
    roc_lncrna_disease_matrix = new_lncrna_disease_matrix + lncrna_disease_matrix

    train_matrix_file = str(i) + "times_need_lncran_disease_tr.h5"
    with h5py.File(train_matrix_file, 'w') as hf:
        hf.create_dataset("rating", data=new_lncrna_disease_matrix)

    rel_matrix = new_lncrna_disease_matrix
    row_n = rel_matrix.shape[0]
    col_n = rel_matrix.shape[1]
    temp_l = np.zeros((row_n, row_n))
    temp_d = np.zeros((col_n, col_n))
    adj = np.vstack((np.hstack((temp_l, rel_matrix)), np.hstack((rel_matrix.T, temp_d))))

    modeldir = "D:\\model"
    bestout = None
    bestloss = np.inf
    x = np.arange(col_n, col_n + row_n)
    y = np.arange(col_n)
    for epoch in range(1, 1000):
        out, loss = methods.train(epoch, rel_matrix, encoder, decoder, None, device, gen_optimizer, lncx, disx, adj, None, x, y)
        print('the '+str(epoch)+' times loss is'+str(loss))
        if bestloss > loss:
            bestloss = loss

    print(bestloss)
    output = out.cpu().data.numpy()

    score_matrix = output

    aa = score_matrix.shape
    bb = roc_lncrna_disease_matrix.shape
    zero_matrix = np.zeros((score_matrix.shape[0], score_matrix.shape[1])).astype('int64')
    print(score_matrix.shape)
    print(roc_lncrna_disease_matrix.shape)

    score_matrix_temp = score_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_lncrna_disease_matrix == 2)] = minvalue - 10
    sorted_lncrna_disease_matrix, sorted_score_Matrix = sortscore.sort_matrix(score_matrix, roc_lncrna_disease_matrix)

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    for cutoff in range(sorted_lncrna_disease_matrix.shape[0]):
        P_matrix = sorted_lncrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_lncrna_disease_matrix[cutoff + 1:sorted_lncrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)

        accuracy_list.append(accuracy)
    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    all_recall.append(recall_list)
    all_precision.append(precision_list)
    all_accuracy.append(accuracy_list)
tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)
accuracy_arr = np.array(all_accuracy)

mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
mean_cross_fpr = np.mean(fpr_arr, axis=0)
mean_cross_recall = np.mean(recall_arr, axis=0)
mean_cross_precision = np.mean(precision_arr, axis=0)
mean_cross_accuracy = np.mean(accuracy_arr,axis=0)
file = open('mean_cross_recall.txt', 'w')
for i in mean_cross_recall:
    file.write(str(i) + '\n')
file.close()
file = open('mean_cross_precision.txt', 'w')
for i in mean_cross_precision:
    file.write(str(i) + '\n')
file.close()
file = open('mean_cross_tpr.txt', 'w')
for i in mean_cross_tpr:
    file.write(str(i) + "\n")
file.close()

file = open('mean_cross_fpr.txt', 'w')
for i in mean_cross_fpr:
    file.write(str(i) + "\n")
file.close()
file=open('mean_cross_accuracy.txt','w')
for i in mean_cross_accuracy:
      file.write(str(i)+"\n")
file.close()

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)

plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)  # 要加上这一句才能显示label
plt.savefig("roc3.png")
print("runtime over, now is :")
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()













