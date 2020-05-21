# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:07:11 2020

@author: Real
"""
import numpy as np
from cnn_utils import *
np.random.seed(1)
def prepare_data():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # index = 6
    # plt.imshow(X_train_orig[index])
    # print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    # print ("number of training examples = " + str(X_train.shape[0]))
    # print ("number of test examples = " + str(X_test.shape[0]))
    # print ("X_train shape: " + str(X_train.shape))
    # print ("Y_train shape: " + str(Y_train.shape))
    # print ("X_test shape: " + str(X_test.shape))
    # print ("Y_test shape: " + str(Y_test.shape))
    # conv_layers = {}
    return X_train, Y_train,X_test, Y_test,classes