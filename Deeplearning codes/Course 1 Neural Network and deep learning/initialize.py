# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:51:29 2020

@author: Real
"""
import numpy as np
def initialize_with_zeros(dim):
    ### START CODE HERE ### ( 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b