# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 22:14:35 2018

@author: yuguangyang
"""

import numpy as np
from sklearn . model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


iris = datasets.load_iris()
print(iris .data . shape)
print(iris .target. shape)


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

model = svm.SVC(kernel='linear',C=1)
model .fit(X_train,y_train)
res = model.score(X_test, y_test)

cv_res = cross_val_score(model, iris.data, iris.target, cv=5)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
resList = list()
X = iris.data
y = iris.target
for train, test in kf.split(iris.data):
    print("%s %s" % (train, test)) #print out the index
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    model. fit(X_train, y_train)
    resList.append(model.score(X_test, y_test))

cv_res = cross_val_score(model, iris.data, iris.target, cv=kf.split(iris.data))

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train, test in ss.split(iris.data):
    print("%s %s" % (train, test)) #print out the index
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    model .fit(X_train,y_train)
    resList.append(model.score(X_test, y_test))
    
cvres = cross_val_score(model, iris.data, iris.target, cv=ss.split(iris.data))    
    
from sklearn . model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test)) #print out the index
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    model .fit(X_train,y_train)
    resList.append(model.score(X_test, y_test))
    
    
cv_res = cross_val_score(model, iris.data, iris.target, cv=skf.split(X, y))
from sklearn . model_selection import StratifiedShuffleSplit
skf = StratifiedShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test)) #prnt out the -index
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    model.fit(X_train,y_train)
    resList.append(model.score(X_test, y_test))

cv_res = cross_val_score(model, iris.data, iris.target, cv=skf.split(X, y))
    