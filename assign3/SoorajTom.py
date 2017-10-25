# -*- coding: utf-8 -*-
"""
@author: sooorajt
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, KernelPCA
import random
import itertools

def makeshell(r,n, d):
    
    xs = []
    ys = []
    zs = []
    
    for i in range(n):
        rad = random.uniform(r, r + d)
        xt = random.uniform(-1,1)
        yt = random.uniform(-1,1)
        zt = random.uniform(-1,1)
        l = math.sqrt(xt*xt + yt*yt + zt*zt)
        xs.append((xt/l) * rad)
        ys.append((yt/l) * rad)
        zs.append((zt/l) * rad)
    
    return (xs, ys, zs)

def getslice(xs, ys, zs, d):
    xp = []
    yp = []
    for i in range(len(xs)):
        if(abs(zs[i]) <= d):
            xp.append(xs[i])
            yp.append(ys[i])
    
    return xp, yp
    
def plot3D(xs, ys, zs, labs, title):
    
    clsx = [[],[],[]]
    clsy = [[],[],[]]
    clsz = [[],[],[]]
    
    for i in range(len(xs)):
#    ax3.scatter(xs[i], ys[i], zs[i], c=col, marker='o')
            clsx[labs[i]].append(xs[i])
            clsy[labs[i]].append(ys[i])
            clsz[labs[i]].append(zs[i])
            
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    
    for c,i in [('r',0),('g', 1), ('b', 2)]:
        ax3.scatter(clsx[i], clsy[i], clsz[i], c=c, marker='o')
    
    ax3.set_title(title)
    ax3.set_xlabel('X Label')
    ax3.set_ylabel('Y Label')
    ax3.set_zlabel('Z Label')

def doKmeans(xs, ys, zs = [0], plot = 1, title= "K-means Clustering"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]

    dataset = zip(xs,ys,zs)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset)
    
    clsx = [[],[],[]]
    clsy = [[],[],[]]
    clsz = [[],[],[]]
    
    for i in range(len(dataset)):
#    ax3.scatter(xs[i], ys[i], zs[i], c=col, marker='o')
            clsx[kmeans.labels_[i]].append(xs[i])
            clsy[kmeans.labels_[i]].append(ys[i])
            clsz[kmeans.labels_[i]].append(zs[i])
    if(plot == 1):
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')
        
        for c,i in [('r',0),('g', 1), ('b', 2)]:
            ax3.scatter(clsx[i], clsy[i], clsz[i], c=c, marker='o')
        
        ax3.set_title(title)
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
    return (kmeans.labels_)

def doSpectral(xs, ys, zs = [0], plot = 1, title= "Spectral Clustering"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]
    
    dataset = zip(xs,ys,zs)
    
#    X = StandardScaler().fit_transform(dataset)
    
    specclus = SpectralClustering(n_clusters=3, gamma=1.0, n_jobs=-1).fit(dataset)
    
#    specclus = SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=10.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0,  degree=3, coef0=1, kernel_params=None, n_jobs=4).fit(dataset)
    
    clsx = [[],[],[]]
    clsy = [[],[],[]]
    clsz = [[],[],[]]
    
    for i in range(len(dataset)):
        clsx[specclus.labels_[i]].append(dataset[i][0])
        clsy[specclus.labels_[i]].append(dataset[i][1])
        clsz[specclus.labels_[i]].append(dataset[i][2])
        
    if(plot == 1):
        fig4 = plt.figure()
        ax3 = fig4.add_subplot(111, projection='3d')
            
        for c,i in [('r',0),('g', 1), ('b', 2)]:
            ax3.scatter(clsx[i], clsy[i], clsz[i], c=c, marker='o')
        
        ax3.set_title(title)
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
    return (specclus.labels_)
    
def doGMM(xs, ys, zs = [0], plot = 1, title= "GMM"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]
        
    dataset = zip(xs, ys, zs)
    
    classifier = GMM(n_components=3, covariance_type='full', init_params='wc', n_iter=20)
    classifier.fit(dataset)
    res = classifier.predict(dataset)
    
    clsx = [[],[],[]]
    clsy = [[],[],[]]
    clsz = [[],[],[]]
    
    for i in range(len(dataset)):
        clsx[res[i]].append(dataset[i][0])
        clsy[res[i]].append(dataset[i][1])
        clsz[res[i]].append(dataset[i][2])
    
    if(plot == 1):
        fig4 = plt.figure()
        ax3 = fig4.add_subplot(111, projection='3d')
            
        for c,i in [('r',0),('g', 1), ('b', 2)]:
            ax3.scatter(clsx[i], clsy[i], clsz[i], c=c, marker='o')
        
        ax3.set_title('GMM')
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
    return (res)

def doPCA(xs, ys, zs, n1, n2, n3):
    dataset = zip(xs, ys, zs)
    
    pca = PCA(n_components=2)
    res = pca.fit_transform(dataset)
    
    fig, ax5 = plt.subplots()
    fig, ax6 = plt.subplots()
    
    ds1 = res[0:n1]
    ds2 = res[n1:n2 + n1]
    ds3 = res[n1 + n2:]
    
    ax5.set_title('PCA Datapoints 2D')
    ax5.scatter([item[0] for item in ds1], [item[1] for item in ds1], c = 'r', marker = 'o' )
    ax5.scatter([item[0] for item in ds2], [item[1] for item in ds2], c = 'g', marker = 'o' )
    ax5.scatter([item[0] for item in ds3], [item[1] for item in ds3], c = 'b', marker = 'o' )
    
    labels = doKmeans([item[0] for item in res], [item[1] for item in res], 0)
    
    res1 = PCA(n_components=1).fit_transform(res)
    
    ds1 = res1[0:n1]
    ds2 = res1[n1:n2 + n1]
    ds3 = res1[n1 + n2:]
    
    ax6.set_title('Kernel PCA Datapoints 1D')
    ax6.scatter([item[0] for item in ds3], [0 for i in range(n3)], c = 'b', marker = '+' )
    ax6.scatter([item[0] for item in ds2], [0 for i in range(n2)], c = 'g', marker = '+' )
    ax6.scatter([item[0] for item in ds1], [0 for i in range(n1)], c = 'r', marker = '+' )
    
    doKmeans([item[0] for item in res1], [0 for item in range(len(res))])
    
def doKPCA(xs, ys, zs, n1, n2, n3):
    dataset  = zip(xs, ys, zs)
    
    kpca= KernelPCA(n_components = 2, kernel = 'poly', gamma = 1.0, degree=2, coef0 = 1, n_jobs=-1)
    
    res = kpca.fit_transform(dataset)
    
    fig, ax5 = plt.subplots()
    
    ds1 = res[0:n1]
    ds2 = res[n1:n2 + n1]
    ds3 = res[n1 + n2:]
    
    ax5.set_title('Kernel PCA Datapoints 2D')
    
    ax5.scatter([item[0] for item in ds1], [item[1] for item in ds1], c = 'r', marker = '.' )
    ax5.scatter([item[0] for item in ds2], [item[1] for item in ds2], c = 'g', marker = '.' )
    ax5.scatter([item[0] for item in ds3], [item[1] for item in ds3], c = 'b', marker = '.' )
    
    kpca1 = KernelPCA(n_components = 1, kernel = 'poly', gamma = 1.0, degree=2, coef0 = 1, n_jobs=-1)
    
    res = kpca1.fit_transform(dataset)
    
    fig, ax6 = plt.subplots()
    
    ds1 = res[0:n1]
    ds2 = res[n1:n2 + n1]
    ds3 = res[n1 + n2:]
    
    ax6.set_title('Kernel PCA Datapoints 1D')
    
    ax6.scatter([item[0] for item in ds3], [0 for i in range(n3)], c = 'b', marker = '+' )
    ax6.scatter([item[0] for item in ds2], [0 for i in range(n2)], c = 'g', marker = '+' )
    ax6.scatter([item[0] for item in ds1], [0 for i in range(n1)], c = 'r', marker = '+' )
    
def main():
    baser = 1
    n1 = 100
    n2 = 600
    n3 = 1200
    xs1, ys1, zs1 = makeshell(baser, n1, .5)
    xs2, ys2, zs2 = makeshell(4 * baser, n2, .5)
    xs3, ys3, zs3 = makeshell(7 * baser, n3, .5)
    
    combinedx = list(itertools.chain(xs1, xs2, xs3))
    combinedy = list(itertools.chain(ys1, ys2, ys3))
    combinedz = list(itertools.chain(zs1, zs2, zs3))
    
    fig1 = plt.figure()
    ax3 = fig1.add_subplot(111, projection='3d')
    ax3.scatter(xs1, ys1, zs1, c='b', marker='^')
    ax3.scatter(xs2, ys2, zs2, c='g', marker='+')
    ax3.scatter(xs3, ys3, zs3, c='r', marker='o')
    
    ax3.set_xlabel('X Label')
    ax3.set_ylabel('Y Label')
    ax3.set_zlabel('Z Label')
    ax3.set_title("Datapoints")

#    xslice, yslice = getslice(combinedx, combinedy, combinedz, baser / 2) 
#    doSpectral(xslice, yslice, [0 for i in range(len(xslice))])
#    
    doKmeans(combinedx, combinedy, combinedz)
    doSpectral(combinedx, combinedy, combinedz)
    doGMM(combinedx, combinedy, combinedz)   
    doPCA(combinedx, combinedy, combinedz, n1, n2, n3)
    doKPCA(combinedx, combinedy, combinedz, n1, n2, n3)
    
    plt.show()
    
if __name__ == '__main__':
    main()