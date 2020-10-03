# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:13:22 2020

@author: yangy
"""

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

import seaborn as sns

plt.close('all')

sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.close('all')
nSample = 10

x = np.linspace(-5, 5, nSample)

y = 2 * x + 1

err = np.random.randn(nSample)

plt.figure(1, figsize=(14, 7))
plt.subplot(121)
plt.plot(x, y + err, '*', label='observation')
plt.plot(x, y, '-', label='model',lw=2)
plt.xlabel('x')
plt.ylabel('y')

x1 = np.linspace(-5, 5, nSample)
x2 = np.linspace(-5, 5, nSample)
X1, X2 = np.meshgrid(x1, x2)


Z = 3 * X1 + X2 + 4
err = np.random.randn(*Z.shape) * 3

ax = plt.subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.scatter(X1, X2, Z + err, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.savefig('linearRegressionModelDemo.png',dpi=300)