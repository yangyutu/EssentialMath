# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:08:47 2020

@author: yangy
"""

# QQ Plot
from numpy.random import seed
from numpy.random import randn
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
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
# seed the random number generator
plt.close('all')
seed(1)
plt.figure(1, figsize=(14, 14))
nSamples = 500
# generate univariate observations
data = 5 * randn(nSamples) + 50
# q-q plot
ax = plt.subplot(221)
qqplot(data, line='s', ax=ax)
plt.title('Normal',fontsize=20, fontweight='bold')

loc, scale = 0., 4.
data = np.random.laplace(loc, scale, nSamples)

ax = plt.subplot(222)
qqplot(data, line='s', ax=ax)

plt.title('Laplace',fontsize=20, fontweight='bold')

data = np.random.uniform(-1, 1, nSamples)
ax = plt.subplot(223)
qqplot(data, line='s', ax=ax)

plt.title('Uniform',fontsize=20, fontweight='bold')

data = np.random.lognormal(1, 1, nSamples)
ax = plt.subplot(224)
qqplot(data, line='s', ax=ax)
plt.title('Lognormal',fontsize=20, fontweight='bold')
plt.tight_layout() 

plt.savefig('QQPlotNormalityTest.png',dpi=300)