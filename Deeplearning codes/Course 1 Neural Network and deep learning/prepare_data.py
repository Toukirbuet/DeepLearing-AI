# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:24:10 2020

@author: Real
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from lr_utils import load_dataset
def prepare_data():
    get_ipython().magic('matplotlib inline')
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    #index = 0
    #plt.imshow(train_set_x_orig[index])
    #print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + " picture.'")
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    train_set_x_flatten =train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten =test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    return train_set_x,train_set_y,test_set_x,test_set_y,num_px,classes
