import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

plt.close('all')
np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch']



plot_num = 1

datasets = [noisy_circles, noisy_moons, aniso, varied]

epsList = [0.1, 0.2, 0.4]

for eps in epsList:
    plt.figure(figsize=(6, 6))
    plt.title(r'$epsilon$=' + str(eps))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
    
        n_cluster = 2 if i_dataset < 2 else 3
        spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=eps)
        
        algorithm = dbscan
        y_pred = algorithm.fit_predict(X)
    
        # plot
        plt.subplot(2, 2, i_dataset + 1)
    
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    
        #centers = algorithm.
        #center_colors = colors[:len(centers)]
        #plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
    
    plt.show()
    plt.savefig('DBSCAN_demo_eps' + str(eps) + '.png', dpi=300)