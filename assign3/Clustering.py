# -*- coding: utf-8 -*-
"""
@author: sooorajt
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
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

def getslice(xs, ys, zs, d, n1, n2, n3):
    xp = []
    yp = []
    m1, m2, m3 = 0,0,0
    for i in range(len(xs)):
        if(abs(zs[i]) <= d):
            xp.append(xs[i])
            yp.append(ys[i])
            if(i < n1):
                m1 += 1
            elif(i < n1 + n2):
                m2 += 1
            else:
                m3 += 1
    
    return xp, yp, m1, m2, m3
    
def plot3D(xs, ys, zs, labs, title):
    
    clsx = [[],[],[]]
    clsy = [[],[],[]]
    clsz = [[],[],[]]
    
    for i in range(len(xs)):
            clsx[labs[i]].append(xs[i])
            clsy[labs[i]].append(ys[i])
            clsz[labs[i]].append(zs[i])
            
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    
    for c,i in [('b',2),('g', 1), ('r', 0)]:
        ax3.scatter(clsx[i], clsy[i], clsz[i], c=c, marker='o')
    
    ax3.set_title(title)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    fig3.savefig(title)

def plot2D(ds, n1, n2, n3, title):
    
    fig, ax3 = plt.subplots()
    
    ax3.set_title(title)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    
    s1 = ds[:n1]
    s2 = ds[n1:n1+n2]
    s3 = ds[n1+n2:]
    
    ax3.scatter([item[0] for item in s3], [item[1] for item in s3], c='r', marker='o')
    ax3.scatter([item[0] for item in s2], [item[1] for item in s2], c='g', marker='o')
    ax3.scatter([item[0] for item in s1], [item[1] for item in s1], c='b', marker='o')
    
def doKmeans(xs, ys, zs = [0], plot = 1, title= "K-means Clustering"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]

    dataset = zip(xs,ys,zs)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset)
    
    if(plot == 1):
        plot3D(xs, ys, zs, kmeans.labels_, title)
    return (kmeans.labels_)

def doSpectral(xs, ys, zs = [0], plot = 1, title= "Spectral Clustering"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]
    
    dataset = zip(xs,ys,zs)
    
    specclus = SpectralClustering(n_clusters=3, gamma=1.0, n_jobs=1).fit(dataset)
    
    if(plot == 1):
        plot3D(xs, ys, zs, specclus.labels_, title)
    return (specclus.labels_)
    
def doGMM(xs, ys, zs = [0], plot = 1, title= "GMM"):
    if(zs == [0]):
        zs = [0 for i in range(len(xs))]
        
    dataset = zip(xs, ys, zs)
    
    classifier = GMM(n_components=3, covariance_type='full', init_params='wc', n_iter=20)
    classifier.fit(dataset)
    res = classifier.predict(dataset)
    
    if(plot == 1):
        plot3D(xs, ys, zs, res, title)
    return (res)

def docluster(xs, ys, zs, xc, yc, zc, tsuffix = ""):
    kmlab = doKmeans(xc, yc, zc, plot=0)
    plot3D(xs, ys, zs, kmlab, "K-means " + tsuffix)
    
    sclab = doSpectral(xc, yc, zc, plot=0)
    plot3D(xs, ys, zs, sclab, "Spectral clustering " + tsuffix)
    
    gmmlab = doGMM(xc, yc, zc, plot=0)
    plot3D(xs, ys, zs, gmmlab, "GMM " + tsuffix)

def doPCA(xs, ys, zs, n1, n2, n3):
    dataset = zip(xs, ys, zs)
    labels = [0 for i in range(n1)] + [1 for i in range(n2)] + [2 for i in range(n3)]

    res = PCA(n_components=2).fit_transform(dataset)
    plot3D([item[0] for item in res], [item[1] for item in res], [0 for item in res], labels, 'PCA Datapoints 2D')
    
    plot2D(res, n1, n2, n3, "Datapoints after PCA - 2D")
    
    docluster(xs, ys, zs, [item[0] for item in res], [item[1] for item in res], [0 for item in res], "after PCA to 2D")
    
    res1 = PCA(n_components=1).fit_transform(res)
    plot3D([item[0] for item in res1], [0 for item in res1], [0 for item in res1], labels, 'PCA Datapoints 1D')
    
    plot2D(zip([item[0] for item in res1],[0 for item in res1]), n1, n2, n3, "Datapoints after PCA - 1D")
    
    docluster(xs, ys, zs, [item[0] for item in res1], [0 for item in res1], [0 for item in res1], "after PCA to 1D")
    
def doKPCA(xs, ys, zs, n1, n2, n3):
    dataset  = zip(xs, ys, zs)
    
    kpca= KernelPCA(n_components = 2, kernel = 'poly', gamma = 1.0, degree=2, coef0 = 1, n_jobs=-1)
    res = kpca.fit_transform(dataset)
    
    labels = [0 for i in range(n1)] + [1 for i in range(n2)] + [2 for i in range(n3)]
    plot3D([item[0] for item in res], [item[1] for item in res], [0 for item in res], labels, 'Kernel PCA Datapoints 2D')
    
    plot2D(res, n1, n2, n3, "Datapoints after KPCA - 2D")
    
    docluster(xs, ys, zs, [item[0] for item in res], [item[1] for item in res], [0 for item in res], "after Kernel-PCA to 2D")
    
    kpca1 = KernelPCA(n_components = 1, kernel = 'poly', gamma = 1.0, degree=2, coef0 = 1, n_jobs=-1)
    res1 = kpca1.fit_transform(dataset)
    
    plot3D([item[0] for item in res1], [0 for item in res1], [0 for item in res1], labels, 'Kernel PCA Datapoints 1D')
    plot2D(zip([item[0] for item in res1],[0 for item in res1]), n1, n2, n3, "Datapoints after KPCA - 1D")
    
    docluster(xs, ys, zs, [item[0] for item in res1], [0 for item in res1], [0 for item in res1], "after Kernel-PCA to 1D")
    
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
    ax3.scatter(xs1, ys1, zs1, c='b', marker='o')
    ax3.scatter(xs2, ys2, zs2, c='g', marker='o')
    ax3.scatter(xs3, ys3, zs3, c='r', marker='o')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("Datapoints")

    xslice, yslice, m1, m2, m3 = getslice(combinedx, combinedy, combinedz, baser, n1, n2, n3) 
    
    plot2D(zip(xslice, yslice), m1, m2, m3, "Sliced view")
#    doSpectral(xslice, yslice, [0 for i in range(len(xslice))], title="Sliced view")
#    
#    doKmeans(combinedx, combinedy, combinedz)
#    doSpectral(combinedx, combinedy, combinedz)
#    doGMM(combinedx, combinedy, combinedz)   
#    docluster(combinedx, combinedy, combinedz, combinedx, combinedy, combinedz)
#    doPCA(combinedx, combinedy, combinedz, n1, n2, n3)
#    doKPCA(combinedx, combinedy, combinedz, n1, n2, n3)
    
    plt.show()
    
if __name__ == '__main__':
    main()