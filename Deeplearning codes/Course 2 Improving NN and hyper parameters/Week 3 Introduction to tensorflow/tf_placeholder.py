# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:31:05 2020

@author: Real
"""

# Change the value of x in the feed_dict
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_eager_execution()
#matplotlib inline
np.random.seed(1)
sess = tf.Session()
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()