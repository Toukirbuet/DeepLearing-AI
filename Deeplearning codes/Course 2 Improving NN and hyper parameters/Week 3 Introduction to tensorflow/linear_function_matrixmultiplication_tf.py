# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:34:52 2020

@author: Real
"""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
#tf.disable_eager_execution()

# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
     ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X") ##None
    W = tf.constant(np.random.randn(4,3), name = "W") ##None
    b = tf.constant(np.random.randn(4,1), name = "b") ##NoneX=tf.placeholder()
    Y=tf.add(tf.matmul(W,X),b)
    sess=tf.Session()
    result=sess.run(Y)
    sess.close()
    return result
print( "result = \n" + str(linear_function()))
