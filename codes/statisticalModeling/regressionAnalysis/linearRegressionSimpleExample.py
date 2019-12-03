# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:04:09 2018

@author: yuguangyang
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import plotFormat

plotFormat.set_pubAll()

nobs = 100
X = np.linspace(0,1,100)

e = np.random.normal(0,0.1,nobs)
#e = e*(2*X)

X = sm.add_constant(X)
beta = [2,1]
y = np.dot(X,beta) +e
results = sm.OLS(y,X).fit()
print(results.summary())
fig = plt.figure(1)
plt.plot(X[:,1],y, 1 * 1 )
plt.show()
## test for normaLity of residuaLs

sm.qqplot(results.resid)
import pylab
import scipy.stats as scipystats
scipystats.probplot(results.resid, dist="norm", plot=pylab)
pylab.show()