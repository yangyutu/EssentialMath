# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:23:04 2020

@author: yangy
"""
from numpy.random import seed
from numpy.random import randn
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.stats import norm
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
plt.close('all')
np.random.seed(1)
x = np.random.randn(20)

mu_set = np.linspace(-10, 10, 100)
sd_set = [.5, 1, 2, 3, 5]
max_val = max_val_location = None
plt.figure(figsize=(9,8))
for i in sd_set:
    ll_array = []
    
    for j in mu_set:
        temp_mm = 0
        
        for k in x:
            temp_mm += np.log(norm.pdf(k, j, i)) # The LL function
        ll_array.append(temp_mm)
    
        if (max_val is None):
            max_val = max(ll_array)
        elif max(ll_array) > max_val:
            max_val = max(ll_array)
            max_val_location = j
        
    # Plot the results
    plt.plot(mu_set, ll_array, label="sd: %.1f" % i)
    
    print("The max LL for sd %.2f is %.2f" % (i, max(ll_array)))    
    plt.axvline(x=0, color='black', ls='-.')
    plt.legend(loc='upper left')


plt.xlabel(r"mean $\mu$")
plt.ylabel("Log Likelihood")
plt.ylim(-100, -30)
plt.savefig('log_likelihood_normal_demo1.png', dpi=300)





mu = 0
sd = 1
mu_set = np.linspace(-10, 10, 100)
for i in range(5):
    ll_array = []
    n = (i + 1) * 10
    x = np.random.randn(n)
    for mu in mu_set:

        
        plt.figure(2, figsize=(9,8))
        
        
        temp_mm = 0
        
        for k in x:
            temp_mm += np.log(norm.pdf(k, mu, sd)) # The LL function
        ll_array.append(temp_mm)

    # Plot the results
    plt.plot(mu_set, ll_array, label="n: %d" % n)
    
    #print("The max LL for sd %.2f is %.2f" % (i, max(ll_array)))    
    plt.axvline(x=0, color='black', ls='-.')
    plt.legend(loc='upper left')
    
plt.ylim(-300, -10)
plt.xlabel(r"mean $\mu$")
plt.ylabel("Log Likelihood")
plt.savefig('log_likelihood_normal_demo2.png', dpi=300)