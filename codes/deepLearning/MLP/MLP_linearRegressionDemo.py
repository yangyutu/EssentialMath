#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:30:13 2019

@author: yangyutu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib as mpl
import seaborn as sns
plt.close('all')
sns.set()
mpl.rcParams.update({'font.size': 20})
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

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

import torch
import torch.nn as nn


np.random.seed(1)
plt.close('all')

x = np.linspace(-1, 1, 40)


X = np.c_[x.ravel(), (x**2).ravel(), (x**3).ravel(), (x**4).ravel(), (x**5).ravel(), (x**6).ravel(), (x**7).ravel(), (x**8).ravel(), (x**9).ravel(), (x**10).ravel()]

y_raw = 0.3 + 2 * x + 0.5 * x**2 - 1.6 * x**3
y = y_raw + 0.3 * np.random.randn(x.shape[0])


input_size_list = [1, 3, 6, 10]

for idx, input_size in enumerate(input_size_list):
    output_size = 1
    # Linear regression model
    model = nn.Linear(input_size, output_size)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    nStep = 300
    
    X_torch = torch.from_numpy(X[:,0:input_size]).float()
    y_torch = torch.from_numpy(y).float()
    for i in range(nStep):
        y_pred = model(X_torch).squeeze()
        optimizer.zero_grad()
        loss = criterion(y_torch, y_pred)
        loss.backward()
        optimizer.step()
        
    y_pred = model(X_torch).squeeze().detach().numpy()
    
    
    plt.figure(2, figsize=(14, 14))
    plt.subplot(2,2,idx+1)
    plt.plot(x, y, 'o', markerfacecolor='none')
    plt.plot(x, y_raw)
    plt.plot(x, y_pred)
    plt.xlabel('x')
    plt.ylabel('y')

plt.figure(2, figsize=(14, 14))
plt.subplot(2,2,1)
plt.legend(['sample','true','prediction'])
plt.savefig('MLP_linearRegressionDemo.png',dpi=300)