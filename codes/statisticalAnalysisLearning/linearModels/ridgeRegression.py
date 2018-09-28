# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 08:15:25 2018

@author: yuguangyang
"""

from sklearn.linear_model import Ridge
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
lamb = 1
clf = Ridge(alpha=lamb)
clf.fit(X, y) 
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)

beta = clf.coef_
beta0 = clf.intercept_

# testing the result via analytical solution

IdentitySize = X.shape[1]
IdentityMatrix= np.identity(IdentitySize)
XtX_lamb = X.T.dot(X) + lamb * IdentityMatrix
XtY = X.T.dot(y)
res = np.linalg.solve(XtX_lamb, XtY);
test = XtX_lamb.dot(res) - XtY     