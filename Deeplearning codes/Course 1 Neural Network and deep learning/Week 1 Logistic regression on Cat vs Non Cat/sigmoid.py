# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:19:40 2020

@author: Real
"""

import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-x)))
x=np.array([0,1])
print(sigmoid(x))