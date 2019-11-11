# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:36:38 2018

@author: yuguangyang
"""
from __future__ import print_function
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
np.random.seed(1024)

nsample = 200
x = np.linspace(0, 20, nsample)
X = sm.add_constant(x)
beta = [5., 0.5]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6//10:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e 
X = X[:,[0,1]]

plt.plot(y,x,'o')
plt.xlabel('x')
plt.ylabel('y')

mod_wls = sm.WLS(y, X, weights=1./(w ** 2))
res_wls = mod_wls.fit()
print(res_wls.summary())

res_ols = sm.OLS(y, X).fit()
print(res_ols.params)
print(res_ols.summary())