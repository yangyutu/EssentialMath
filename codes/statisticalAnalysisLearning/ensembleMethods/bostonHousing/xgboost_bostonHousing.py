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
import xgboost as xgb
import numpy as np
import matplotlib as mpl
plt.close('all')

boston = load_boston()


print(boston.keys())

X = boston.data
y = boston.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 3, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train, y_train)


y_pred = xg_reg.predict(X_test)



print("RMSE: {0}".format(np.sqrt(mean_squared_error(y_test, y_pred))))


params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 4, 'alpha': 10}

data_matrix = xgb.DMatrix(data=X, label=y)

cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3, num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


cv_results.head()

print((cv_results["test-rmse-mean"]).tail(1))


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
