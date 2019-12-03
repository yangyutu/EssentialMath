# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:29:59 2018

@author: yuguangyang
"""

# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
#from plotFormat import set_pub, polish
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns
from sklearn import datasets
import matplotlib as mpl
mpl.rcParams.update({'font.size': 24})
sns.set()
plt.close('all')
#set_pub()
# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# now demean such that each column sum up to 0

sd = StandardScaler()
X = sd.fit_transform(X)

y = np.array(range(10)).astype(np.float)

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X = sd.fit_transform(X)  # Standardize data (easier to set the l1_ratio parameter)

#y = normalize(y.reshape(-1, 1)).flatten()

# #############################################################################
# Compute paths L2

n_alphas = 200
alphas = np.logspace(-3, 5, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
    
# #############################################################################
# Display results

fig = plt.figure(1, figsize=(6,5))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'coefficients, $\beta$')
ax.set_title('Ridge regression path')
plt.axis('tight')
plt.show()
fig.savefig('RidgePath.png', format='png', dpi=300)
#polish(ax)

# #############################################################################
# Compute paths Elastic
n_alphas = 200
alphas = np.logspace(-3, 5, n_alphas)

coefs = []
for a in alphas:
    lasso = linear_model.Lasso(alpha=a, fit_intercept=False)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)
    
    

# #############################################################################
# Display results

fig = plt.figure(2, figsize=(6,5))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'coefficients, $\beta$')
ax.set_title('Lasso regression path')

plt.show()
fig.savefig('LassoPath.png', format='png', dpi=300)

# #############################################################################
# Compute paths L1
n_alphas = 200
alphas = np.logspace(-3, 5, n_alphas)

coefs = []
for a in alphas:
    ene = linear_model.ElasticNet(alpha=a,  l1_ratio=0.5, fit_intercept=False)
    ene.fit(X, y)
    coefs.append(ene.coef_)
    
    

# #############################################################################
# Display results

fig = plt.figure(3, figsize=(6,5))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'coefficients, $\beta$')
ax.set_title('Elastic Net regression path')
plt.axis('tight')
plt.show()
fig.savefig('ElasticPath.png', format='png', dpi=300)