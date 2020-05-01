# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:54:18 2020

@author: Real
"""

# GRADED FUNCTION: compute_cost
import numpy as np

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    ### END CODE HERE ###

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

#testcase
'''
def compute_cost_test_case(): 
    Y = np.asarray([[1, 1, 1]]) 
    aL = np.array([[.8,.9,0.4]]) 
    return Y, aL
Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))
'''