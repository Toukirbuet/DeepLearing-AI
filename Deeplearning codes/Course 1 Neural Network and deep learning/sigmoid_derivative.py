# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:24:02 2020

@author: Real
"""

import numpy as np
def sigmoid_derivative(x):
    return (1/(1+np.exp(-x)))
x=np.array([0,1])
p=sigmoid_derivative(x)*(1-sigmoid_derivative(x))
print(p)