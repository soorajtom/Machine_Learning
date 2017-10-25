# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:04:38 2017

@author: labs
"""
import numpy as np
from numpy import genfromtxt as gft
import matplotlib.pyplot as plt


def leastSquareRegression(X, Y) :
    Xt = X.transpose()
    inverse = np.linalg.inv(Xt * X)
    w = inverse * Xt * Y
    return w

def ridgeRegression(X, Y, lamb):
    I = np.matrix(np.eye(X.shape[1]))
    Xt = X.transpose()
    inverse = np.linalg.inv(Xt * X + lamb * I)
    w = inverse * Xt * Y
    return w

def makeFeature(fMatrix, num = 10) :
    biglist = list()
    for x in fMatrix :
       lst = [1]
       for i in range (num) :
           lst.append(lst[-1] * x)
       biglist.append(lst)
    fmatrix = np.matrix(biglist)
    return fmatrix

def ridgeRegressionStochastic(X, Y, lamb, alpha = 0.001) :
    #set (assume) reqd. constants
    w = np.matrix([1.0 for i in range (X.shape[1])]).transpose()
    eps = 0.000001    
    diff = 10
    error1 = Y.transpose() * Y
    print X.shape
    print Y.shape
    while diff > eps :
        for i in range (X.shape[0]) :
            xx = X[i, :].reshape(X.shape[1], 1)
            yy = Y[i, :].reshape(1, 1)
            # do the update
            sub = alpha * (xx * (w.transpose() * xx - yy) + lamb * w)
            w = w - sub
        error = Y - X * w
        error2 = error.transpose() * error
        diff = abs(error1 - error2)
        error1 = error2
    return w

def crossValidate(X, Y, Xts, Yts, lamb, alpha = 0.001) :
#    print 'X\'s shape = ', X.shape, 'Y\'s shape = ', Y.shape, type(X)
    #split data set into 5 pieces
    A = np.split(X[0:X.shape[0] - X.shape[0] % 5, :], 5)
    B = np.split(Y[0:Y.shape[0] - Y.shape[0] % 5, :], 5)
    
    error = [0.0, 0.0, 0.0]
    for i in range (5) :
        C = np.zeros(shape=(0,X.shape[1]))
        D = np.zeros(shape=(0, Y.shape[1]))
        #keep ith block as validation set.
        for j in range (5) :
            #create new training set C
            if j != i :
                C = np.vstack((C, A[j]))
                D = np.vstack((D, B[j]))
        
        #call ridge Regression with new training set C , D
        wrg = ridgeRegressionStochastic(C, D, lamb, alpha)
        
        #calculate error for this model on the validation set.
        #validation set is A[i], B[i]
        err = B[i] - A[i] * wrg
        error[1] += err.transpose() * err #square error
        #calculate training error
        err = D - C * wrg
        error[0] += err.transpose() * err
        #calculate testing error
        err = Yts - Xts * wrg
        error[2] += err.transpose() * err
    #take avg. of 5 errors. for each kind of error
    error[0] /= 5
    error[1] /= 5
    error[2] /= 5
    return error

def selectHyperParam(X, Y, Xts, Yts, alpha = 0.001):
     #list of lambdas
    lst = [2**(-10)]
    for i in range (20) :
        lst.append(lst[-1] * 2)
    
    #initialize list of errors
    errList = []
    #do cross validation for training data for each lambda
    for lamb in lst :
        errList.append(crossValidate(X, Y, Xts, Yts, lamb, alpha))
    
    #optimal lambda is the one with min. error
    errList2 = errList
    errList2.sort()
    #call function to plot graphs
    plotErrorGraphs(lst, errList)
    #find min lambda
    lambIndex = errList.index(errList2[0])
    minLamb = lst[lambIndex]
    return minLamb
    
def plotErrorGraphs(lst, errList) :
    print 'errorlist', errList[0][0][0,0]
    
    Y1 = list()
    Y2 = list()
    Y3 = list()
    for i in errList :
        Y1.append(i[0][0,0])
        Y2.append(i[1][0,0])
        Y3.append(i[2][0,0])
    print Y1
    #plot graphs
    fig, ax = plt.subplots()
    ax.plot(lst, Y1, 'ro-', label = 'training error')
    ax.plot(lst, Y2, 'go-', label = 'validation error')
    ax.plot(lst, Y3, 'bo-', label = 'test error')
    plt.xlabel(r'$log \lambda$')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('log square error')
    plt.title(r'$\lambda\hspace{0.5}vs\hspace{0.5}square\hspace{0.5}error$')
    legend = ax.legend(loc=0, shadow = True)
    fig.savefig('errors.jpg')
    plt.show()
    
def main() : 
    #read data from files
    x = gft('./newRegressiondata/x.txt')
    y = gft('./newRegressiondata/y.txt')
    xts = gft('./newRegressiondata/xts.txt')
    yts = gft('./newRegressiondata/yts.txt')
#    x = [[1,1,1,1], [1,2,3,4], [2,3,4,1], [3,4,1,2], [4,1,2,3], [4,3,2,1]]
#    y = [5,12,13,14,11,13]
    
    #reshape
    y = y.reshape(y.size, 1)
    yts = yts.reshape(yts.size, 1)
    
    if len(x.shape) == 1 :
        x = x.reshape(x.size, 1)
    if len(xts.shape) == 1 :
        xts = xts.reshape(xts.size, 1)
    
    x = np.matrix(x)
    Y = np.matrix(y)
    Yts = np.matrix(yts)
    xts = np.matrix(xts)
    #make features for x
    X = makeFeature(x, num = 5)
    Xts = makeFeature(xts, num = 5)
    
    lambOptimum = selectHyperParam(X, Y, Xts, yts)
    print 'Optimum lambda is ', lambOptimum
    
    #try plotting data and ridge regression value for optimum lambda
    fig, ax = plt.subplots()
    ax.plot(X, Y, 'ro', label = 'Training data')
    ax.plot(X, X * (ridgeRegressionStochastic(X, Y, lambOptimum)), 'yo', label = 'training data ridge regression')
    ax.plot(Xts, Yts, 'g^', label = 'Test data')
    ax.plot(xts, xts * (ridgeRegressionStochastic(Xts, Yts, lambOptimum)), 'b^', label = 'test data ridge regression')
    plt.xlabel('input data')
    legend = ax.legend(loc = 0)
    plt.show()
#    fMat = makeFeature(X)
#    print 'fMat = \n', fMat
#    
#    wls = leastSquareRegression(fMat, Y)
#    print 'wls = ', wls
#    
#    #call ridge regression using stochastic gd.
#    wrgs = ridgeRegressionStochastic(fMat, Y, 0.0)
#    print 'wrgs = ', wrgs
#    
##    plt.plot(X, Y, 'ro', X, fMat * wls, 'bo', X, fMat * wrgs, 'go')
#    fig, ax = plt.subplots()
#    ax.plot(X, Y, 'ro', label = 'Training data')
#    ax.plot(X, fMat * wls, 'bs', label = 'least square regression')
#    ax.plot(X, fMat * wrgs, 'g^', label  = 'ridge regression' )
#    legend = ax.legend(loc = 0, shadow=True)
#    plt.xlabel('X')
#    plt.ylabel('f(X)')
#    fig.suptitle('graph')
#    fig.savefig('regressions.jpg')
#    plt.show()
    
    
main()    