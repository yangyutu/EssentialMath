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
from sklearn.preprocessing import StandardScaler
plt.close('all')
#set_pub()
# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# now demean such that each column sum up to 0

sd = StandardScaler()
X = sd.fit_transform(X)

y = np.array(range(10))

# #############################################################################
# Compute paths L1

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
    
###############################################################################
# using the analytical solution method
# https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress    
IdentitySize = X.shape[1]
IdentityMatrix= np.identity(IdentitySize)


coefs2 = []
for a in alphas:
    lamb = a
    XtX_lamb = X.T.dot(X) + lamb * IdentityMatrix
    XtY = X.T.dot(y)
    res = np.linalg.solve(XtX_lamb, XtY);
    coefs2.append(res)

test = XtX_lamb.dot(res) - XtY 
test2 = XtX_lamb.dot(ridge.coef_) - XtY 
# #############################################################################
# Display results

fig = plt.figure(1, figsize=(6,6))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'coefficients, $\beta$')
ax.set_title('Ridge regression Path')
plt.axis('tight')
plt.show()
fig.savefig('RidgePath.png', format='png', dpi=300)
#polish(ax)

# #############################################################################
# Compute paths L1
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
    
###############################################################################
# using the analytical solution method
# https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress    
IdentitySize = X.shape[1]
IdentityMatrix= np.identity(IdentitySize)


coefs2 = []
for a in alphas:
    lamb = a
    XtX_lamb = X.T.dot(X) + lamb * IdentityMatrix
    XtY = X.T.dot(y)
    res = np.linalg.solve(XtX_lamb, XtY);
    coefs2.append(res)

test = XtX_lamb.dot(res) - XtY 
test2 = XtX_lamb.dot(ridge.coef_) - XtY 
# #############################################################################
# Display results

fig = plt.figure(1, figsize=(6,6))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'coefficients, $\beta$')
ax.set_title('Ridge regression Path')
plt.axis('tight')
plt.show()
fig.savefig('LassoPath.png', format='png', dpi=300)