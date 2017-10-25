import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn import mixture
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier as RFC

def genData(r=10,eps=0.1,n=1000):
    theta = np.random.uniform(0,2*np.pi,n)
    phi = np.arccos(np.random.uniform(0,2,n)-1)
    radius = np.random.uniform(r-eps,r+eps,n)
    rsin_phi = radius * np.sin(phi)
    x = rsin_phi * np.cos(theta)
    y = rsin_phi * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x,y,z))
    
X1=genData(1,0.5,100)
X2=genData(4,0.5,400)
X3=genData(9,0.5,900)
X=np.row_stack((X1,X2,X3))
c=[0]*100+[1]*400+[2]*900

#--------------------------
spectral = cluster.SpectralClustering(n_clusters=3, affinity="rbf",n_jobs=-1,gamma=1.0)
spectralfit=spectral.fit(X)
spectralpred =spectralfit.labels_.astype(np.int)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter( xs= X[:,0] , ys = X[:,1], zs=X[:,2], s=10, c=spectralpred)
ax.set_title('Spectral Clustering')


#---------------------
kmeans = cluster.KMeans(n_clusters=3, random_state=0)
kMeansfit = kmeans.fit(X)
kMeanspred = kmeans.labels_.astype(np.int)
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection='3d')
ax1.scatter( xs= X[:,0] , ys = X[:,1], zs=X[:,2], s=10, c=kMeanspred)
ax1.set_title('K-Means Clustering')


gmm = mixture.GaussianMixture(n_components=3, covariance_type='full')
gmmfit = gmm.fit(X)
gmmpred =gmm.predict(X)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter( xs= X[:,0] , ys = X[:,1], zs=X[:,2], s=10, c=gmmpred)
ax3.set_title('GMM')


pca = decomposition.PCA(n_components=2)
P = pca.fit_transform(X)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter( xs= P[:,0] , ys = P[:,1], s=10, c=c)
ax4.set_title('PCA in 2 dimension')


def my_kernel(X, Y):
    return np.square(np.dot(X, Y.T)+1)

kpca = decomposition.KernelPCA(kernel="precomputed", gamma=10)
KP = kpca.fit_transform(my_kernel(X,X))
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter( xs= KP[:,0] , ys = KP[:,1], s=10, c=c)
ax5.set_title('KPCA in 2 dimension')

data = np.genfromtxt('breastcancer.txt',delimiter=',')
np.random.shuffle(data)
n,d=data.shape
split = round(0.6*n)
train, test = data[:split,:], data[split:,:]
randomForest = RFC(10,oob_score=True,max_depth=5)

plt.show()

