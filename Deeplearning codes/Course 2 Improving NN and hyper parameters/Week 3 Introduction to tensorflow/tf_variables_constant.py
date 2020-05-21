# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:28:56 2020

@author: Real
"""
'''
"Session()" has been removed with TF 2.0.
I inserted two lines. 
One is tf.compat.v1.disable_eager_execution() 
and the other is sess = tf.compat.v1.Session() # instead of tf.Session()
tf.compat.v1--> converts tensorflow 2.0 to 1.x
'''
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
tf.disable_eager_execution()

#matplotlib inline
np.random.seed(1)

#constant
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
sess = tf.Session()
print(sess.run(c))

#variable
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39
loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss)) 