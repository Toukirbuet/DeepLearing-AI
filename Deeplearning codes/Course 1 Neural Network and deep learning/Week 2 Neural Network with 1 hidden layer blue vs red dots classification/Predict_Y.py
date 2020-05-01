# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:20:17 2020

@author: Real
"""
import numpy as np
from Forward_Prop import forward_propagation

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in
    ,→ X
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue:
    ,→ 1)
    """
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### ( 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    ### END CODE HERE ###
    return predictions