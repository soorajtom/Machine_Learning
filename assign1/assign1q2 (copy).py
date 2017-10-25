#assignment 1 Q2
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt as genft

def LeastSquares(fMatrix, y):
	Xt = np.matrix(fMatrix)
	Y = np.matrix(y)
	Yt = Y.transpose()
	X = Xt.transpose()
	inverse = np.linalg.inv(Xt * X)
	w = inverse * Xt * Yt
	return w

def makeFeature(inMat, n):
    fList = []
    for x in inMat:
        frow = [1]
        for i in range(n - 1):
            frow.append(frow[i] * x)
        fList.append(frow)
    fList = np.matrix(fList)
    return fList

#def RidgeRegression(X, Y, lamb):
#    I = np.matrix(np.eye(X.shape[1]))
#    Xt = X.transpose()
#    inverse = np.linalg.inv(Xt * X + lamb * I)
#    w = inverse * Xt * Y
#    return w

def RidgeRegressionStochastic(X, y, lamb):
    alpha = 0.001
    eps = 0.000001
    diff = 10.0
    y = np.matrix(y).transpose()
    X = np.matrix(X).transpose()
    w = [0 for i in range(X.shape[1])]
    w = np.matrix(w)
    err1 = y.transpose() * y
    while diff > eps:
        for x in range(X.shape[0]):
            w = w - alpha * (X[x,] * (w.transpose() * X[x,] - y[x]) + lamb * w)
        err2 = (y - X * w).transpose() * (y - X * w)
        diff = abs(err2 - err1)
        err1 = err2
    return w

    #w - aplha * xi * wtran (xi - yi) - aplha * lambda * w
################################
#def aneRidgeRegressionStochastic(X, Y, lamb, alpha = 0.001) :
#    #set (assume) reqd. constants
#    Y = np.matrix(Y).transpose()
#    X = np.matrix(X).transpose()
#    w = np.matrix([1.0 for i in range (X.shape[1])]).transpose()
#    eps = 0.000001    
#    diff = 10
#    error1 = Y.transpose() * Y
#    while diff > eps :
#        for i in range (X.shape[0]) :
#            xx = X[i, :].reshape(X.shape[1], 1)
#            yy = Y[i, :].reshape(1, 1)
#            # do the update
#            sub = alpha * (xx * (w.transpose() * xx - yy) + lamb * w)
#            w = w - sub
#        error = Y - X * w
#        error2 = error.transpose() * error
#        diff = abs(error1 - error2)
#        error1 = error2
#    return w
#####################3333

def Crossvalidator5(X, y, lamb):
    sqerrva = 0.0
    sqerrtr = 0.0
    sqerrte = 0.0
    testX = genft('/home/labs/mac_ler/assign1/newRegressiondata/xts.txt')
    testY = genft('/home/labs/mac_ler/assign1/newRegressiondata/yts.txt')
    #reshape
    testX = testX.reshape(testX.size, 1)
    testY = testY.reshape(testY.size, 1)
    
    for i in range(5):
        tempX = X[0:i*(X.shape[0]/5)]
        tempX = np.append(tempX, X[(i+1)*(X.shape[0]/5) : X.shape[0]])
        
        tempy = y[0:i*(y.shape[0]/5)]
        tempy = np.append(tempy, y[(i+1)*(y.shape[0]/5) : y.shape[0]])
        
        tempw = RidgeRegressionStochastic(tempX, tempy, lamb)
        
        for j in xrange(i * (X.shape[0]/5) , (i + 1) * (X.shape[0]/5), 1):
            sqerrva = sqerrva + (y[j] - X[j] * tempw)**2
        for k in xrange(0, i * (X.shape[0]/5), 1):
            sqerrtr = sqerrtr + (y[k] - X[k] * tempw)**2
        for k in xrange((i+1) * (X.shape[0]/5), X.shape[0], 1):
            sqerrtr = sqerrtr + (y[k] - X[k] * tempw)**2
        for m in xrange(0, testX.shape[0], 1):
            sqerrte = sqerrte + (testY[k] - testX[k] * tempw)**2
    err = [(sqerrva)/X.shape[0], sqerrtr/((X.shape[0] - 5) * 5), sqerrte/testX.shape[0]]
    
    return err

def main():
    inMat = genft('/home/labs/mac_ler/assign1/newRegressiondata/x.txt')
    y = genft('/home/labs/mac_ler/assign1/newRegressiondata/y.txt')
    print LeastSquares(inMat, y)
    print RidgeRegressionStochastic(inMat, y, 0.5)
    
    lamb = 2**-11
    validerr = []
    trainerr = []
    testerr = []
    for j in range(21):
        lamb = lamb * 2
        print j
        validerr.append(math.log(float(Crossvalidator5(inMat, y, lamb)[0])))
        trainerr.append(math.log(float(Crossvalidator5(inMat, y, lamb)[1])))
        testerr.append(math.log(float(Crossvalidator5(inMat, y, lamb)[2])))
    print validerr
    
    
    xaxis = [i for i in xrange(-10, 11, 1)]
    fig, ax = plt.subplots()
    ax.plot(xaxis, validerr, 'go-', label= 'Validation error')
    ax.plot(xaxis, trainerr, 'ro-', label= 'Training error')
    ax.plot(xaxis, testerr, 'bo-' , label= 'Testing error')
    plt.xlabel(r'$log \lambda$')
    plt.ylabel('log (square error)')
    plt.title(r'$\lambda\hspace{0.5}vs\hspace{0.5}square\hspace{0.5}error$')
    ax.legend(loc=0, shadow = True)
    fig.savefig('plot.png')
    plt.show()

if __name__ == "__main__":
    main()