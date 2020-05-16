# -*- coding: utf-8 -*-
"""
Created on Sun May 17 01:42:49 2020

@author: Real
"""
'''
"Session()" has been removed with TF 2.0.
I inserted two lines. 
One is tf.compat.v1.disable_eager_execution() 
and the other is sess = tf.compat.v1.Session() # instead of tf.Session()
'''
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
tf.compat.v1.disable_eager_execution()

#matplotlib inline
np.random.seed(1)

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
sess = tf.compat.v1.Session()
print(sess.run(c))