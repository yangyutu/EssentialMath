#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:31:32 2019

@author: yangyutu
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

plt.close('all')
n_colors = 5

# Load the Summer Palace photo
china = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

fig = plt.figure(1, figsize=(12,12))
plt.subplot(221)
plt.title('original')
plt.imshow(china)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


n_colors = [2, 5, 64]

for idx, color in enumerate(n_colors):
    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(china.shape)
    assert d == 3
    image_array = np.reshape(china, (w * h, d))
    
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=color, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))
    
    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))
    
    plt.subplot(2,2, idx+2)
    plt.axis('off')
    plt.title('Segmentation: K=' + str(color))
    plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))