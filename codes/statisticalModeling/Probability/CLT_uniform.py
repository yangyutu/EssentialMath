# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:17:25 2020

@author: yangy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import seaborn as sns

sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.close('all')




def random_samples(size):
    ''' Return a list of random variates distributed
        according to the log-normal distribution.'''
    # define parameters of our log-normal distribution
    rv = np.random.uniform(-5, 5, size=size)
    return rv
 
N = 1000000
mean_sizes_m = [1, 10, 100]
 
rv = random_samples(N)
rv_split = [[rv[i:i+m] for i in range(0, len(rv), m)]
            for m in mean_sizes_m]
sample_means = [[s.mean() for s in samples]
                for samples in rv_split]


x = np.linspace(-7, 7, 100)
colors = ['orange', 'r', 'c']
plt.figure(1, figsize=[9,9]) 
# plot the histograms
for i, set_of_sample_means in enumerate(sample_means):
    lab = 'n = {} sample mean'.format(mean_sizes_m[i])
    plt.hist(set_of_sample_means, bins=x,
             normed=True, alpha=0.5,
             color=colors[i],
             label=lab)
 
# define parameters of our log-normal distribution
sig, mu = 1, 3
 
# plot our distribution
lognorm_pdf = [sp.stats.uniform.pdf(x=i, loc=-5, scale=10)
               for i in x]

plt.plot(x, lognorm_pdf, lw=4, c='b', alpha=0.5,
         label='probability distribution')
 
# plot the mean of the distribution,
plt.axvline(0, 0, 1,
            color='black', alpha=0.5, label='mean')
 
plt.title('Sample means from uniformly distributed samples'
          .format(N), y=1.03, fontsize=20)
plt.ylabel('Normalized frequency', labelpad=18)
plt.xlabel('$x$', labelpad=18)
 
plt.legend()
plt.savefig('uniform_central_limit_theroem.png',
             bbox_inches='tight', dpi=300)