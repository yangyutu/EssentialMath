# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:15:29 2020

@author: yangy
"""

import numpy as np
import statsmodels.api as sm
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rc('font', size=18)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=20)
plt.rc('axes', titlesize=20)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

plt.close('all')

nsample = 100
x = np.linspace(0, 10, 100)


########## Positive correlation####################

beta = np.array([1, 3, 10])
e = np.random.normal(size=nsample)

z = np.zeros_like(e)

z[0] = e[0]
rho = 0.8
for i in range(1, len(z)):
    z[i] = rho * z[i - 1] + e[i]
    



y = x + z

plt.figure(1, figsize=(12, 6))
ax = plt.subplot(121)

ax.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()

ax.plot(x, results.fittedvalues)


ax = plt.subplot(122)
ax.plot(x, y - results.fittedvalues, '-o')
plt.xlabel('x')
plt.ylabel('Residuel')

plt.savefig('linearRegressionAutoRegressionErrorDemoPostive.png', dpi=300)
print(results.summary())



##########Negative correlation####################

z = np.zeros_like(e)

z[0] = e[0]
rho = -0.8
for i in range(1, len(z)):
    z[i] = rho * z[i - 1] + e[i]
    



y = x + z

plt.figure(2, figsize=(12, 6))
ax = plt.subplot(121)

ax.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()

ax.plot(x, results.fittedvalues)


ax = plt.subplot(122)
ax.plot(x, y - results.fittedvalues, '-o')
plt.xlabel('x')
plt.ylabel('Residuel')
plt.savefig('linearRegressionAutoRegressionErrorDemoNegative.png', dpi=300)
print(results.summary())