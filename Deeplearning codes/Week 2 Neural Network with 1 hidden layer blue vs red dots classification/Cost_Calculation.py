# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:55:38 2020

@author: Real
"""
import numpy as np

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (1 Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1,
    ,→ number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1,
    ,→ W2 and b2
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1] # number of example
    # Compute the cross-entropy cost
    ### START CODE HERE ### ( 2 lines of code)
    logprobs = np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost = - np.sum(logprobs)/m
    ### END CODE HERE ###
    # use directly np.dot())
    # cost=-(np.dot(Y,np.log(A2.T))+np.dot(np.log(1-A2),(1-Y).T))/m
    cost = np.squeeze(cost) # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    return cost