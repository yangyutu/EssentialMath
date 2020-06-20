# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:04:09 2018

@author: yuguangyang
"""

# stackoverflow.com/questions/30011618/statsmodels-summary-to-latex
import numpy as np
import statsmodels.api as sm
import math
import pandas as pd
from linearRegressionPlotUtils import diagnostic_plots
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=18)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=20)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, np.sqrt(x)))
beta = np.array([1, 3, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()


print(results.summary())

plt.close('all')
diagnostic_plots(pd.DataFrame(X), pd.DataFrame(y), figSize=(7,7))
print(results.summary().as_latex())

plt.figure(1)
plt.savefig('SimpleLRDiagnosis-residual.png',dpi=300)
plt.figure(2)
plt.savefig('SimpleLRDiagnosis-QQ.png',dpi=300)
plt.figure(3)
plt.savefig('SimpleLRDiagnosis-scaleLocation.png',dpi=300)
plt.figure(4)
plt.savefig('SimpleLRDiagnosis-cookdistance.png',dpi=300)