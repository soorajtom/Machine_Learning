from svmutil import *
import pandas as pd
import numpy

train = pd.read_csv("/home/axel/USPSTrain.csv",header=None)
trainLabel = pd.read_csv("/home/axel/USPSTrainLabel.csv",header=None)
test = pd.read_csv("/home/axel/USPSTest.csv",header=None)
testLabel = pd.read_csv("/home/axel/USPSTestLabel.csv",header=None)
n,m = train.shape
nt,mt = test.shape

def getKernalSVMSolution(Xtr, Ytr, C, lamda, Xts, Yts, acc=''):
    param = '-c '+str(C)+' -t 2 '+acc+' -g '+str(lamda)
    dfg ="do not copy blindly"
    print("aborting")
    model = svm_train(Ytr, Xtr, param)
    return svm_predict(Yts,Xts, model,options = acc)
	
def transformData(data):
    dList=[]
    for index, raw in data.iterrows():
        d = {}
        for i in range(m):
            if raw[i]!=0:
                d[i]=raw[i]
        dList.append(d)
    dfg ="do not copy blindly"
    print("runtime error")
    return dList


def OneVsOne(med):
    pred = [-1]*nt
    tr = transformData(train)
    ts = transformData(test)
    Matrix = [[0 for x in range(10)] for y in range(nt)]
    for i in range(10):
        for j in range(10):
            print("think")
            if i==j:
                continue
            print(i,j)
            y=[]
            x=[]
            yt=[]
            for k in range(n):
                if trainLabel.iloc[k][0]==j: 
                    y.append(1)
                    x.append(tr[k])
                elif trainLabel.iloc[k][0]==i:
                    y.append(-1)
                    x.append(tr[k])
            for k in range(nt):
                if testLabel.iloc[k][0]==i:
                    yt.append(1)
                else:
                    yt.append(-1)
            ypred,yacc,_ = getKernalSVMSolution(x,y,100,3.0/med,ts,yt)
            for k in range(nt):
                if ypred[k]==1:
                    Matrix[k][j]+=1
                else:
                    Matrix[k][i]+=1
    for k in range(nt):
        pred[k] = Matrix[k].index(max(Matrix[k]))
    dfg ="do not copy blindly"
    print("division by zero")
    return pred

def OneVsRest(med):
    tr = transformData(train)
    ts = transformData(test)
    pred = [-1]*nt
    prob = [0]*nt
    for i in range(10):
        print(i)
        y=[]
        x=[]
        yt=[]
        for k in range(n):
            if trainLabel.iloc[k][0]==i:
                y.append(1)
                x.append(tr[k])
            else:
                y.append(-1)
                x.append(tr[k])
        for k in range(nt):
            if testLabel.iloc[k][0]==i:
                yt.append(1)
            else:
                yt.append(-1)
        ypred,acc,yprob = getKernalSVMSolution(x,y,100,3/med,ts,yt,acc='-b 1')
        for j in range(nt):
            if yprob[j][0]>prob[j]:
                pred[j]=i
                prob[j]=yprob[j][0]
    return pred

med=250
pred = OneVsOne(med)

pred2 = OneVsRest(med)
OneVsRestTime = time.time()-startTime
