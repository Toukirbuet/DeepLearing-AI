# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:13:55 2020

@author: Real
"""
# import tensorflow.compat.v1 as tf #
# tf.disable_v2_behavior() # v2
import tensorflow as tf

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(shape=[None,n_H0,n_W0,n_C0],dtype=tf.float32)
    Y = tf.placeholder(shape=[None,n_y],dtype=tf.float32)
    ### END CODE HERE ###
    
    return X, Y

# X, Y = create_placeholders(64, 64, 3, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))
