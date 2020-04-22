# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:27:00 2020

@author: Real
"""
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
def Logistic_Regression(X,Y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    
    
    LR_predictions = clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' %float((np.dot(Y,LR_predictions) +np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +'% ' + "(percentage of correctly labelled datapoints)")