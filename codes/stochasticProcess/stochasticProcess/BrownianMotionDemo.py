# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:26:12 2020

@author: yangy
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.subplots_adjust(wspace =0.5, hspace =0.5)
plt.close('all')
np.random.seed(1)
x0 = 0
nTraj = 3
nStep = 1000
dt = 0.1
plt.figure(1, figsize=(8, 8))
for _ in range(nTraj):
    
    xSet = [x0]
    tSet = [0]
    x = x0
    for i in range(nStep):
        x += np.random.randn() * np.sqrt(dt)
        xSet.append(x)
        tSet.append((i + 1) * dt)
    plt.plot(tSet, xSet, lw = 2)
    

    plt.xlabel(r'$t$', fontsize=24)
    plt.ylabel(r'$W(t)$', fontsize=24)
    
plt.savefig('BronwianMotionDemo.png', dpi=300)