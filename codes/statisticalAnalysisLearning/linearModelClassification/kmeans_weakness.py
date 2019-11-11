#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:06:07 2019

@author: yangyutu
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.close('all')

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples,centers=4, random_state=random_state)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
# Turn off tick labels
plt.xticks([])
plt.yticks([])
# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")
# Turn off tick labels
plt.xticks([])
plt.yticks([])
# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[0.5, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")
plt.xticks([])
plt.yticks([])

# bad initials
fixedCenters = np.array([[-1, -1], [-1, 1], [1, -1], [2, 2]]) * 4
X, y = make_blobs(n_samples=n_samples,centers=fixedCenters, random_state=random_state)

initCenters = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1]]) * 0.0

y_pred = KMeans(n_clusters=4, init = initCenters, random_state=random_state).fit_predict(X)

plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("bad initial cluster centers")
plt.xticks([])
plt.yticks([])


plt.figure(2, figsize=(12,6))
plt.subplot(121)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Kmeans")
plt.xticks([])
plt.yticks([])
y_pred = KMeans(n_clusters=4, init = 'k-means++', random_state=random_state).fit_predict(X)

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("KMeans ++")
plt.xticks([])
plt.yticks([])