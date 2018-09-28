# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 17:31:33 2018

@author: yuguangyang
"""

print(__doc__)

# Author: Gilles Louppe <g.louppe@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import plotFormat

plotFormat.set_pubAll()
plt.close('all')
# Settings
n_repeat = 50       # Number of iterations for computing expectations
n_train = 5000        # Size of the training set
n_test = 1000       # Size of the test set
noise = 0.1         # Standard deviation of the noise
np.random.seed(0)

# Change this for exploring the bias-variance decomposition of other
# estimators. This should work well for estimators with high variance (e.g.,
# decision trees or KNN), but poorly for estimators with low variance (e.g.,
# linear models)
# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.sin(x)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y


X_train, y_train = generate(n_samples=n_train, noise=noise, n_repeat=1)
X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=1)


baggingSize = [1,5,10,20,30,50,100,500,1000]

err = []
y_predict_set = []

for size in baggingSize:
    bagRegressor = BaggingRegressor(DecisionTreeRegressor(),n_estimators = size)
    bagRegressor.fit(X_train, y_train)
    y_predict = bagRegressor.predict(X_test)
    err.append(mean_squared_error(y_predict, y_test))
    y_predict_set.append(y_predict)
    
    


plt.close('all')
plt.figure(1)
plt.tight_layout()
plt.plot(X_test,y_test)
plt.plot(X_test,y_predict_set[0])
plt.xlabel('x',fontweight='bold')
plt.ylabel('y',fontweight='bold')
plt.legend(['Test Data','BaggingSize=1'])

plt.figure(2)
plt.tight_layout()
plt.plot(X_test,y_test)
plt.plot(X_test,y_predict_set[-1])
plt.xlabel('x',fontweight='bold')
plt.ylabel('y',fontweight='bold')
plt.legend(['Test Data','BaggingSize=1000'])

y_true = f(X_test)
intrin_error = mean_squared_error(y_test, y_true)

plt.figure(3)
plt.tight_layout()
plt.plot(baggingSize, err,marker='o',color='black')
plt.xlabel('Bagging Size',fontweight='bold')
plt.ylabel('Mean Squared Error',fontweight='bold')
plt.ylim([0.01,0.025])

plt.show()