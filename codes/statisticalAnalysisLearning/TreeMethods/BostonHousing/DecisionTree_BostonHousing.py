#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:50:36 2019

@author: yangyutu
"""


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree
import seaborn as sns

sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.close('all')

boston = load_boston()


print(boston.keys())

X = boston.data
y = boston.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

treeModel = DecisionTreeRegressor()


treeModel.fit(X_train, y_train)


y_pred = treeModel.predict(X_test)



print("RMSE: {0}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

plt.figure(1, figsize=(10,10))
sklearn.tree.plot_tree(treeModel,max_depth=2)
#fig = plt.gcf()
#fig.set_size_inches(150, 100)
#fig.savefig('tree.png', dpi=300)

# Plot feature importance
feature_importance = treeModel.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(2, figsize=(10,10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('decisionTree_variableImportance.png',dvi=300)

