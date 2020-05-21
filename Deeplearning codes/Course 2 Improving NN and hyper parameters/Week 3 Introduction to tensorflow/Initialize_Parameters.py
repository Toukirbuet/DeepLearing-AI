# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:18:06 2020

@author: Real
"""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_v2_behavior()
#matplotlib inline
np.random.seed(1)

# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.truncated_normal_initializer(stddev=0.1)) #as contrib deppreciated in tf 2 so X tf.contrib.layers.xavier_initializer(seed = 1)) ##None
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer()) ##None
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.truncated_normal_initializer(stddev=0.1))#tf.contrib.layers.xavier_initializer(seed = 1)) ##None
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer()) ##None
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.truncated_normal_initializer(stddev=0.1))#tf.contrib.layers.xavier_initializer(seed = 1)) ##None
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer()) ##None
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters