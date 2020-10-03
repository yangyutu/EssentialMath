# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:11:35 2020

@author: yangy
"""

import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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


plt.figure(1, figsize=(15, 7))
x = [(i + 1) * 0.1 for i in range(9)]
y = [norm.ppf(e) for e in x]
plt.subplot(121)
plt.plot(x, y, '-*b')
plt.ylabel('percentile', labelpad=18)
plt.xlabel(r'$\alpha$', labelpad=18)


# plot our distribution


xx = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)
plt.figure(1)
plt.subplot(122)
plt.plot(xx, norm.pdf(xx),
       'r-', lw=2, alpha=0.6)
plt.ylim([0, 0.5])
for e in  y:
    print(norm.pdf(e))
    plt.axvline(e, 0, norm.pdf(e) / 0.5,
                color='black', alpha=0.5)
plt.ylabel('$f(x)$', labelpad=18)
plt.xlabel('$x$', labelpad=18)
plt.savefig('percentile_normal.png',
             bbox_inches='tight', dpi=300)