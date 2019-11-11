#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:00:49 2019

@author: yangyutu
"""


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


index = np.random.choice(X.shape[0], 5000)
X_train = X[index]

n_components = 50

pca = PCA(n_components = n_components)

pca.fit(X_train)

PCs = pca.components_

def plot_gallery(images, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
        
        
plot_gallery(PCs, 28, 28, 5, 10)
plt.savefig('Mnist_eigendigit.png', format='png', dpi=300)
plot_gallery(X_train, 28, 28, 5, 10)
plt.savefig('Mnist_snapshot.png', format='png', dpi=300)