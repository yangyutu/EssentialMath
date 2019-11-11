#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:40:16 2019

@author: yangyutu
"""


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import matplotlib as mpl

import numpy as np

plt.close('all')
mpl.rcParams.update({'font.size': 36})
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
plt.rcParams['xtick.labelsize']=28
plt.rcParams['ytick.labelsize']=28
sns.set()

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


index = np.random.choice(X.shape[0], 3000)
X_train = X[index]
y_train = y[index].astype(np.int)

isoMap = Isomap()
isoMap.fit(X_train)

embed = isoMap.embedding_

distMat = isoMap.dist_matrix_

U, S, V = np.linalg.svd(distMat)

eigenValue = np.sqrt(S)


plt.figure(1, figsize=(6,5))

plt.plot(eigenValue[0:10])
plt.xlabel('k', fontsize=20)
plt.ylabel('eigenvalue', fontsize=20)
plt.savefig('Mnist_isoMapSpectrum.png', format='png', dpi=300)

plt.figure(2, figsize=(10,10))
plt.scatter(embed[:, 0], embed[:, 1], c=y_train, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);

plt.savefig('Mnist_lowEmding.png', format='png', dpi=300)

data = X[y == '5'][0:2000]

isoMap = Isomap()
isoMap.fit(data)

embed = isoMap.embedding_
plt.figure(3, figsize=(10, 10))
ax = plt.gca()
ax.plot(embed[:, 0], embed[:, 1], '.k')
from matplotlib import offsetbox

min_dist_2 = (0.05 * max(embed.max(0) - embed.min(0))) ** 2
shown_images = np.array([2 * embed.max(0)])
for i in range(data.shape[0]):
    dist = np.sum((embed[i] - shown_images) ** 2, 1)
    if np.min(dist) < min_dist_2:
    # don't show points that are too close
        continue
    shown_images = np.vstack([shown_images, embed[i]])
    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(data[i].reshape((28, 28)), cmap=plt.cm.gray), embed[i])
    ax.add_artist(imagebox)
    
plt.savefig('Mnist_lowEmdingAnalysisDigit_5.png', format='png', dpi=300)