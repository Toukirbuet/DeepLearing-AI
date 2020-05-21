# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:57:57 2020

@author: Real
"""

# Change the value of x in the feed_dict
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_eager_execution()
#matplotlib inline
np.random.seed(1)
def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x=tf.placeholder(tf.float32,name='x')
    sigmoid=tf.sigmoid(x)
    sess=tf.Session()
    result=sess.run(sigmoid, feed_dict={x:z})
    sess.close()
    return result

print(sigmoid(0))
    
    