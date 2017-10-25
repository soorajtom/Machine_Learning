# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:00:09 2017

@author: Sooraj Tom
"""

from svmutil import *
import numpy as np
import matplotlib.pylab as plt
import statistics
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score


train = np.genfromtxt("../../USPSTrain.csv", delimiter = ',')
trainLabel = np.genfromtxt("../../USPSTrainLabel.csv", delimiter = ',')
test = np.genfromtxt("../../USPSTest.csv", delimiter = ',')
testLabel = np.genfromtxt("../../USPSTestLabel.csv", delimiter = ',')

trr,trc = train.shape
tsr,tsc = test.shape

def getKernelSVMSolution(Xtr, Ytr, C, lamda, Xts, Yts, param=''):
    opt = "-c " + str(C) + " -g " + str(lamda) + " -t 2 -q " + param
    model = svm_train(Ytr, Xtr, opt)
    return svm_predict(Yts, Xts, model, '-q ' + param)

def genDictList(data):
    rows,cols = data.shape
    dictList = []
    for i in range(rows):
        dic = {}
        for j in range(cols):
            if(data[i][j] != 0):
                dic[j + 1] = data[i][j]
        dictList.append(dic)
    return dictList

def writeToFile(data, filname):
    outfil = open('../../' + filname, "w")
    for i in range(len(data)):
        outfil.write(str(data[i]) + "\n")

def plotimage(data, pred, act, filname):
    fig = plt.figure()
    fig.set_size_inches(16, 16)
    j = 0
    k = 1
    for i in range(len(pred)):
        if(pred[i] != act[i]):
            j += 1
    rows = int((j) ** (0.5) + 1)
    for i in range(data.shape[0]):
        if(pred[i] != act[i]):
            image = (np.matrix(data[i])).reshape(16,16)
            plt.subplot(rows, rows, k)
            plt.title("P "+ str(pred[i]) + " | A "+ str(int(act[i])))
            plt.axis('off')
            k += 1
            plt.imshow(image, interpolation='nearest', cmap=plt.cm.Greys_r ,clim=(0.0, 2.0))
    
    fig.tight_layout()
    fig.savefig(filname)
    
def F1scoreCalc(Ypred, Ylab):
    return f1_score(Ylab, Ypred, average="weighted")
    
def OneVsOne(med):
    traindata = genDictList(train)
    testdata = genDictList(test)
    tdatacons = [[] for i in range(10)]
    
    Pmatrix = [[0 for i in range(10)] for j in range(tsr)]
    confMatrix = [[0 for j in range(10)] for i in range(10)]
    Pred = []
    err = 0

    startt = time.time()
    
    for i in range(trr):
        tdatacons[int(trainLabel[i])].append(traindata[i])
    for i in range(10):
        for j in range(i + 1, 10):
            y = [1 for k in range(len(tdatacons[i]))]
            y+= [-1 for k in range(len(tdatacons[j]))]
            x = tdatacons[i] + tdatacons[j]
            
            (p_labels, p_acc, p_vals) = getKernelSVMSolution(x, y, 100, 3.0/med, testdata, testLabel)
            
            for k in range(len(p_labels)):
                if(p_labels[k] == 1.0):
                    Pmatrix[k][i] += 1
                else:
                    Pmatrix[k][j] += 1
    
    stopt = time.time()
    
    for k in range(tsr):
        Pred.append(Pmatrix[k].index(max(Pmatrix[k])))
        confMatrix[int(Pred[k])][int(testLabel[k])] += 1
            
    print "\n====One vs One======\n"
    print "Confusion matrix (One vs One)"
    print (confMatrix)
    print "\nF1 score(One vs One):" + str(F1scoreCalc(Pred, testLabel))
    print "Time taken: " + str(stopt - startt)
    
    plotimage(test, Pred, testLabel, "../../ErrorsOneVersusOne.png")
    writeToFile(Pred, "PredOneVersusOne.txt")
    
def OneVsRest(med): 
    traindata = genDictList(train)
    testdata = genDictList(test)
    tdatacons = [[] for i in range(10)]
    
    Pred = [[-1,0] for i in range(tsr)]
    confMatrix = [[0 for j in range(10)] for i in range(10)]
    err = 0
    
    for i in range(trr):
        tdatacons[int(trainLabel[i])].append(traindata[i])
        
    startt = time.time()
    
    for i in range(10):
        y = [1 for k in range(len(tdatacons[i]))]
        x = tdatacons[i]
        xB = []
        for j in range (10):
            if (i == j):
                continue
            xB += tdatacons[j]
            y += [-1 for k in range(len(tdatacons[j]))]
            
        (p_labels, p_acc, p_vals) = getKernelSVMSolution(x + xB, y, 100, 3.0/med, testdata, testLabel, "-b 1")
                        
        for k in range(len(p_labels)):
            if(p_vals[k][0] > Pred[k][1]):
                Pred[k][0] = i
                Pred[k][1] = p_vals[k][0]
                
    stopt = time.time()
    
    PredLabels = [Pred[i][0] for i in range(tsr)]
    
    for k in range(tsr):
        confMatrix[int(PredLabels[k])][int(testLabel[k])] += 1
    
    print "\n====One vs Rest======\n"
    print "Confusion matrix (One vs Rest)"
    print (confMatrix)
    print "\nF1 score(One vs Rest): " + str(F1scoreCalc(PredLabels, testLabel))
    print "Time taken: " + str(stopt - startt)
    
    plotimage(test, PredLabels, testLabel, "../../ErrorsOneVersusRest.png")
    writeToFile(PredLabels, "PredOneVersusRest.txt")
    
def findMedian(data):
    dists = []
    data = np.asarray(data)
    dist = cdist(data, data, 'sqeuclidean')
    r,c = dist.shape
    for i in range (r):
        for j in xrange(i+1, r):
            dists.append(dist[i][j])
    med = statistics.median(dists)
    return med
    
            
if __name__=='__main__':
    OneVsOne(findMedian(train))
    OneVsRest(findMedian(train))