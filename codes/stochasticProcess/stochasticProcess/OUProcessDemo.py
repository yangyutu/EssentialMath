# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:53:30 2020

@author: yangy
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.close('all')


np.random.seed(2)
dt = 1.0/360.0

nSteps = 365 * 3

k = 5

mu = 50
sigma = 0.0020 * 10000

s0 = 50

s = np.zeros(nSteps + 1)




kList = [1, 5, 20]


s[0] = s0
plt.figure(2, figsize=(8,8))
for k in kList:
    
    for i in range(1, nSteps + 1):
        s[i] = s[i - 1] * math.exp(-k * dt) + mu * (1 - math.exp(- k * dt)) + sigma * math.sqrt(dt) * np.random.randn()
    plt.figure(2)
    plt.plot(s)
    plt.ylim([0, 100])
    plt.xlabel('t, days', fontsize=24)
    plt.ylabel('x', fontsize=24)
    
plt.legend(['k=1', 'k=5', 'k=20'])
plt.savefig('OUProcessDemo.png', dpi=300)