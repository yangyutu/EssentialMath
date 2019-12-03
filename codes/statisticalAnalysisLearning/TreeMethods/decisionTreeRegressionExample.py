# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:44:54 2018

@author: yuguangyang
"""

print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import plotFormat

plotFormat.set_pubAll()
#
plt.close('all')
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.cos(2*X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X, y)
dot_data = tree.export_graphviz(regr_1, out_file=None,   
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('RegressionTree_maxdepth2') 


regr_2 = DecisionTreeRegressor(max_depth=5)

regr_2.fit(X, y)
dot_data = tree.export_graphviz(regr_2, out_file=None,   
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('RegressionTree_maxdepth5') 




# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="black", label="max_depth=5", linewidth=2)
plt.xlabel("x",fontweight='bold')
plt.ylabel("y",fontweight='bold')
plt.title("Decision Tree Regression",fontweight='bold')
plt.legend()
plt.show()