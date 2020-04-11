# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:00:44 2020

@author: Real
"""
import numpy as np
from sigmoid import sigmoid

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### ( 1 line of code)
    A = sigmoid(np.dot(w.T,X)+b)
    ### END CODE HERE ###
    for i in range(A.shape[1]):
    # Convert probabilities A[0,i] to actual predictions p[0,i]
    ### START CODE HERE ### ( 4 lines of code)
        if A[0][i]<=0.5:
            A[0][i]=0
        else: 
            A[0][i]=1
    Y_prediction=A
    ### END CODE HERE ###
    assert(Y_prediction.shape == (1, m))
    return Y_prediction