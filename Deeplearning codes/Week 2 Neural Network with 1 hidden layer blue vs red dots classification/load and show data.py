# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:52:29 2020

@author: Real
"""

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from Logistic_regression import Logistic_Regression
#matplotlib inline
#get_ipython().magic('matplotlib inline')
np.random.seed(1) # set a seed so that the results are consistent
X, Y = load_planar_dataset()
#print(X[0].shape)
#plt.scatter(X[0, :], X[1, :], c=Y.reshape (400), s=40, cmap=plt.cm.Spectral)
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # training set size
### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

print('Printing result of logistic regression below-----------------------\n\n')
Logistic_Regression(X,Y)

