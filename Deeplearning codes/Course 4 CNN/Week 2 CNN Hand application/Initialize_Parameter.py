# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:49:55 2020

@author: Real
"""
# import tensorflow.compat.v1 as tf #
# tf.disable_v2_behavior() # v2
import tensorflow as tf #v 1.14

def initialize_parameters():

    tf.set_random_seed(1)                            

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0)) #TF 1.x
    #W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.truncated_normal_initializer(stddev=0.1)) #TF 2.x

    #W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))#TF 2.x
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))  #TF 1.x

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))