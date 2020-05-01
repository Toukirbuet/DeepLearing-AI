# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:13:41 2020

@author: Real
"""
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid,load_planar_dataset, load_extra_datasets
######

from NN import nn_model
from Predict_Y import predict

'''
from  layer_sizes import layer_sizes
from initialize_parameters import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from back_propagation import backward_propagation
from update_parameters import update_parameters
from model_nn import nn_model
from predict import predict
'''
#matplotlib inline
np.random.seed(1) # set a seed so that the results are consistent
X, Y = load_planar_dataset()
# Visualize the data:
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000,print_cost=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) +np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
