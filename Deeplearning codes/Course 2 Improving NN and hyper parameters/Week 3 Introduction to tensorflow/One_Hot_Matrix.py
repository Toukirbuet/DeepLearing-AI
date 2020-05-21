# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:10:43 2020

@author: Real
"""

# GRADED FUNCTION: one_hot_matrix
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_eager_execution()
#matplotlib inline
np.random.seed(1)

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C") ##None
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, depth=C, axis=0) ##None
    
    # Create the session (approx. 1 line)
    sess = tf.Session() ##None
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix) ##None
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close() ##None
    
    ### END CODE HERE ###
    
    return one_hot
'''
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = \n" + str(one_hot))
'''