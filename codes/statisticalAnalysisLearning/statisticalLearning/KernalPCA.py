# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:21:12 2018

@author: yuguangyang
"""

from sklearn.datasets import make_moons
from plotFormat import set_pub, polish
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.close('all')
set_pub()
X,y = make_moons(n_samples = 100, random_state=123)

plt.scatter(X[y==0,0], X[y==0, 1],color='red',marker='^')
plt.scatter(X[y==1,0], X[y==1, 1],color='blue',marker='o')
plt.show()
polish(plt.axes())

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X) #the output is n by 2 array, each row the coefficient of the Principle components.


plt.figure(2)
plt.scatter(X_spca[y==0,0], X_spca[y==0, 1],color='red',marker='^')
plt.scatter(X_spca[y==1,0], X_spca[y==1, 1],color='blue',marker='o')
plt.show()


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_kpca = kpca.fit_transform(X)
plt.figure(3)
plt.scatter(X_kpca[y==0,0], X_kpca[y==0, 1],color='red',marker='^')
plt.scatter(X_kpca[y==1,0], X_kpca[y==1, 1],color='blue',marker='o')
plt.show()


## separating concentraic data
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.figure(4)
plt.scatter(X[y==0,0], X[y==0, 1],color='red',marker='^')
plt.scatter(X[y==1,0], X[y==1, 1],color='blue',marker='o')
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X) #the output is n by 2 array, each row the coefficient of the Principle components.


plt.figure(5)
plt.scatter(X_spca[y==0,0], X_spca[y==0, 1],color='red',marker='^')
plt.scatter(X_spca[y==1,0], X_spca[y==1, 1],color='blue',marker='o')
plt.show()

kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_kpca = kpca.fit_transform(X)
plt.figure(6)
plt.scatter(X_kpca[y==0,0], X_kpca[y==0, 1],color='red',marker='^')
plt.scatter(X_kpca[y==1,0], X_kpca[y==1, 1],color='blue',marker='o')
plt.show()
