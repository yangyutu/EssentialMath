import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

plt.close('all')

plt.figure(figsize=(13, 13))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples,centers=4, random_state=random_state)

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(X_aniso)

plt.subplot(221)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropy, Kmeans")

y_pred = GaussianMixture(n_components=4).fit(X_aniso).predict(X_aniso)
plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropy, GMM")



# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[0.5, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance, Kmeans")
y_pred = GaussianMixture(n_components=3).fit(X_varied).predict(X_varied)
plt.subplot(224)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance, GMM")