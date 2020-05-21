# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:38:45 2020

@author: Real
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



np.random.seed(1)
# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b.astype(float)
    ### END CODE HERE ###

    return Z
'''
a_slice_prev = np.random.randn(2, 2, 1)
W = np.random.randn(2, 2, 1)
b = np.random.randn(1, 1, 1)
print("a_slice[0]:",a_slice_prev[0])
print("W[0]:",W[0])
print("b[0]:",b[0])

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
'''