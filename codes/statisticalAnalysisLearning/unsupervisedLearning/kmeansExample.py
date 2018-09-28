# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 22:48:46 2018

@author: yuguangyang
"""

print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.close('all')
plt.figure(1)

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=2)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)


colors = ['red','green']

plt.scatter(X[:, 0], X[:, 1], c=y_pred,cmap=matplotlib.colors.ListedColormap(colors))
plt.title("kmeans for two clusters",fontweight='bold')



plt.show()