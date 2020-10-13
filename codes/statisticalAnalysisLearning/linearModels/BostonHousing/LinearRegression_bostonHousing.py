#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:21:45 2019

@author: yangyutu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import statsmodels.api as sm
sns.set()

plt.close('all')
data = pd.read_csv('../data/BostonHousing.csv')

# both zn and chas are missing more than 70% data, so will remove both these features
data = data.drop(['zn','chas'],axis=1)


X = data.iloc[:,:-1]
y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train) # Fitting our model to the training set
y_pred = model.predict(X_test)


from sklearn import metrics 
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


X = sm.add_constant(X)
model = sm.OLS(y, X)
result = model.fit()
print(result.summary())