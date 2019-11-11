#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:08:28 2019

@author: yangyutu
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

plt.close('all')

x1 = np.linspace(-10, 10, 30)
x2 = np.linspace(-10, 10, 30)
X1, X2 = np.meshgrid(x1, x2)
X = np.c_[X1.ravel(), X2.ravel()]

y = 2 * X[:,0] + X[:, 1] - 0.8 * X[:,0] * X[:, 1] + 0.5 * X[:, 0] * X[:, 0]

y = y + np.random.randn(y.shape[0]) * 10

fig = plt.figure(1,figsize=(14, 7))
ax = plt.subplot(1, 2, 1, projection='3d')

ax.scatter(X[:,0], X[:,1], y, color='red')
ax.view_init(17, -106)
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
X_poly = PolynomialFeatures(degree=2).fit_transform(X)



lr = LinearRegression().fit(X_poly, y)

y_pred = lr.predict(X_poly)

ax = plt.subplot(1, 2, 2, projection='3d')

y_pred.shape = X1.shape
ax.scatter(X[:,0], X[:,1], y, color='red')
ax.plot_surface(X1, X2, y_pred)
ax.view_init(17, -106)
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
fig.savefig('basisRegression_demo.png', format='png', dpi=300)