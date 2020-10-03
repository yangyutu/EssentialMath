# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:41:06 2020

@author: yangy
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
from linearRegressionPlotUtils import diagnostic_plots
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=18)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=20)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

plt.close('all')
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)

# generate OLS model
model = sm.OLS(y, sm.add_constant(X))
model_fit = model.fit()

# create dataframe from X, y for easier plot handling
dataframe = pd.concat([X, y], axis=1)


diagnostic_plots(X, y, figSize=(7,7))

plt.figure(1)
plt.savefig('Diagnosis-residual.png',dpi=300)
plt.figure(2)
plt.savefig('Diagnosis-QQ.png',dpi=300)
plt.figure(3)
plt.savefig('Diagnosis-scaleLocation.png',dpi=300)
plt.figure(4)
plt.savefig('Diagnosis-cookdistance.png',dpi=300)