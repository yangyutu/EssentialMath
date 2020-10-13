# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:17:11 2020

@author: yangy
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.close('all')
fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
x = np.linspace(-2, 2, 500)
y = np.linspace(-1, 1, 1000)

X, Y = np.meshgrid(x, y)

Z = np.square(X) * (4 - 2.1 * np.square(X) + 0.333 * np.power(X, 4)) + X * Y + np.square(Y) * (-4 + 4 * np.square(Y))


# Plot the 3D surface
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm.jet)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
cset = ax.contour(X, Y, Z, zdir='z', offset=-3, cmap=cm.jet)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
ax.set_zlim(-3, 10)

ax.set_xlabel(r'$x_1$', fontsize=20)
ax.set_ylabel(r'$x_2$', fontsize=20)
ax.set_zlabel(r'$z$', fontsize=20)

plt.show()
plt.savefig('Humpbackfunction.png',dpi=300)

