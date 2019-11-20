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


xgb.plot_tree(xg_reg,num_trees=0)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('XGBtree_1.png', dpi=300)


xgb.plot_tree(xg_reg,num_trees=1)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('XGBtree_2.png', dpi=300)
print('done')

plt.rcParams['figure.figsize'] = [10, 10]
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
plt.rcParams['xtick.labelsize']=28
plt.rcParams['ytick.labelsize']=28
mpl.rcParams.update({'font.size': 36})
xgb.plot_importance(xg_reg)

plt.show()
plt.savefig('XGB_importance.png', dpi=300)