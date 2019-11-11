#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:46:35 2019

@author: yangyutu
"""


from scipy import linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib import cm
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 18})
plt.rc('font', family='serif')
plt.close('all')
# #############################################################################



'''Generate 2 Gaussians samples with different covariance matrices'''
n, dim = 300, 2
np.random.seed(0)
C = np.array([[0., -1.], [2.5, .3]]) * 2.
X = np.r_[np.dot(np.random.randn(n, dim), C),
          np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]

X[0:n, :] += np.array([24, 0])

y = np.hstack((-np.ones(n), np.ones(n)))

sc = StandardScaler()
#X = sc.fit_transform(X)

X0, X1 = X[y == -1], X[y == 1]

fig = plt.figure(1, figsize=(6, 6))
ax = plt.subplot(1,1,1)

ax.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
ax.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')





per = Perceptron(alpha = 0.0001, max_iter=1000, tol = 1e-8)

# class 0 and 1 : areas
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()



per.fit(X, y)




w = per.coef_[0]
b = per.intercept_[0]
print(w, b, per.n_iter_)
x_line = np.linspace(x_min, x_max, nx)
y_line = -w[0] / w[1] * x_line - b / w[1]

plt.figure(1)
ax.plot(x_line, y_line, color='green')
plt.ylim([-50, 50])
y_pred = per.predict(X)

print(classification_report(y, y_pred, labels=[1, 2]))
print('accuracy', accuracy_score(y, y_pred))