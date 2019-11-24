from sklearn import metrics
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import methods
from models import Encoder
from models import Decoder
import itertools


hidden1 = 256 #
hidden2 = 64 #128
hidden3 = 32 #16
with h5py.File('lncRNA_feature.h5', 'r') as hf:
    lncx = hf['infor'][:]
    lncx = torch.Tensor(lncx)
with h5py.File('disease_feature.h5', 'r') as hf:
    diseasex = hf['infor'][:]
    disx = torch.Tensor(diseasex)

with h5py.File('lncrna_disease_association.h5', 'r') as hf:
    lncrna_disease_matrix = hf['rating'][:]
    lncrna_disease_matrix_val =  lncrna_disease_matrix.copy()

device = torch.device('cpu')

all_tpr = []
all_fpr = []

all_recall = []
all_precision = []
all_accuracy = []

#denovo start
for i in range(412):
    new_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    roc_lncrna_disease_matrix = lncrna_disease_matrix.copy()
    if ((False in (new_lncrna_disease_matrix[:,i]==0))==False):
        continue
    new_lncrna_disease_matrix[:,i] = 0

    maxsim = 0
    maxindex = 0
    #compute similarity and replace
    a = diseasex[i, :]
    k = 8
    simlist = []
    simdic = {}
    maxsimarr = np.zeros(k)
    maxindexarr = np.zeros(k)

    predinteration = np.zeros(240).transpose()
    for f in range(412):
        if(f != i) :
            b = diseasex[f,:]
            sim = methods.gaussiansim(a, b)
            for simi in range(k):
                if sim > maxsimarr[simi]:
                    maxsimarr[simi] = sim
                    maxindexarr[simi] = f
                    break

    for maxind in maxindexarr:
        temppred = new_lncrna_disease_matrix[:, int(maxind)]
        predinteration += temppred
    predinteration[predinteration>1] = 1
    new_lncrna_disease_matrix[:, i] = predinteration

    print(new_lncrna_disease_matrix.shape)

    encoder = Encoder(lncx.shape[1], disx.shape[1], hidden1, hidden2, hidden3)
    decoder = Decoder(hidden3)
    gen_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.001, weight_decay=0.005)           #0.0005, 2e-5
    discriminator = None
    dis_optimizer = None
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
        out, loss = methods.train(epoch, rel_matrix, encoder, decoder, discriminator, device, gen_optimizer, lncx, disx,
                                  adj, dis_optimizer, x, y)
        print('the ' + str(epoch) + ' times loss is' + str(loss))
        if bestloss > loss:
            bestloss = loss
            # bestout = out

    print(bestloss)
    score_matrix = out.cpu().data.numpy()
    sort_index = np.argsort(-score_matrix[:,i],axis=0)
    sorted_lncrna_disease_row = roc_lncrna_disease_matrix[:,i][sort_index]

    fileName1 = "denovo" + str(i) + "times.txt"
    file = open(fileName1, 'w')

    for p in score_matrix:
        k = '\t'.join([str(j) for j in p])
        file.write(k + "\n")
    file.close()
    tpr_list = []
    fpr_list = []

    recall_list = []
    precision_list = []

    accuracy_list = []
    for cutoff in range(1,241):
        P_vector = sorted_lncrna_disease_row[0:cutoff]
        N_vector = sorted_lncrna_disease_row[cutoff:]
        TP = np.sum(P_vector == 1)
        FP = np.sum(P_vector == 0)
        TN = np.sum(N_vector == 0)
        FN = np.sum(N_vector == 1)
        tpr = TP/(TP+FN)
        fpr = FP/(FP+TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)

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

mean_denovo_recall = np.mean(recall_arr,axis=0)
mean_denovo_precision = np.mean(precision_arr,axis=0)

mean_denovo_tpr = np.mean(tpr_arr,axis=0)
mean_denovo_fpr = np.mean(fpr_arr,axis=0)
mean_denovo_accuracy = np.mean(accuracy_arr,axis=0)
file=open('mean_denovo_tpr.txt','w')
for i in mean_denovo_tpr:
    file.write(str(i)+'\n')
file.close()
file=open('mean_denovo_fpr.txt','w')
for i in mean_denovo_fpr:
    file.write(str(i)+'\n')
file.close()
file = open('mean_denovo_recall.txt', 'w')
for i in mean_denovo_recall:
    file.write(str(i) + '\n')
file.close()
file = open('mean_denovo_precision.txt', 'w')
for i in mean_denovo_precision:
    file.write(str(i) + '\n')
file.close()
file=open('mean_denovo_accuracy.txt','w')
for i in mean_denovo_accuracy:
      file.write(str(i)+"\n")
file.close()

roc_auc = metrics.auc(mean_denovo_fpr, mean_denovo_tpr)
plt.plot(mean_denovo_fpr,mean_denovo_tpr, label='mean ROC=%0.4f'%roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("pr1.png")
plt.show()
