# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:37:47 2018

@author: yuguangyang
"""

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz 
iris = load_iris()

# Decision tree classifier using default setting
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('IrisDecisionTree_default')  

# Decision tree classifier using minimum sample splitting
clf = tree.DecisionTreeClassifier(min_samples_split = 10)
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('IrisDecisionTree_minSamples')  


# Decision tree classifier using entropy
clf = tree.DecisionTreeClassifier(criterion="entropy",min_samples_split = 10)
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render('IrisDecisionTree_minSamplesEntropy')  