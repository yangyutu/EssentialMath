#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:34:38 2019

@author: yangyutu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
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
data = pd.read_csv('../data/BostonHousing.csv')


data.head()

data.info()

# checking missing (i.e., na) values
data.isna().sum()

data.head()
# Generally, NaN or missing values can be in any form like 0, ? or may be written as âmissingâ and 
# in our case, we can see that there are a lot of 0âs, so we can replace them with NaN to calculate
# how much data we are missing.
data.zn.replace(0, np.nan,inplace = True)
data.chas.replace(0, np.nan,inplace=True)

# visualize missing values
plt.figure(1)
msno.matrix(data)

#Percent of data which is not available
print(data.isnull().sum()/len(data) * 100)

# both âZNâ and âCHASâ are missing more than 70% data, so will remove both these features
data = data.drop(['zn','chas'],axis=1)



data.describe()


# visualize y distribution
plt.figure(2)
sns.distplot(data['medv'])

# visualize correlation

corr = data.corr()
# plot the heatmap
plt.figure(3)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

# visualize pair plot
plt.figure(4)
sns.pairplot(data)

#To fit a linear regression model, we select those features which have 
#a high correlation with our target variable MEDV. By looking at the 
#correlation matrix we can see that RM has a strong positive correlation 
#with MEDV(0.7) whereas LSTAT has a high negative correlation with MEDV(-0.74).

# visualize