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
from plotFormat import set_pub, polish
from sklearn import linear_model
from sklearn import preprocessing
plt.close('all')
set_pub()
# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# now demean such that each column sum up to 0
X = preprocessing.scale(X,with_mean=True,with_std=False)

y = np.array(range(10))
y = preprocessing.scale(y,with_mean=True,with_std=False)

# #############################################################################
# Compute paths

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

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
ax.set_xlabel(r'$\lambda$',fontsize=15)
ax.set_ylabel(r'coefficients, $\beta$',fontsize=15,fontweight='bold')
ax.set_title('coefficients in Ridge regression',fontsize=15,fontweight='bold')
plt.axis('tight')
plt.show()
polish(ax)