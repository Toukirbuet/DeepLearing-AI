# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:09:53 2020

@author: Real
"""

import numpy as np
def normalize(x):
    x_norm=np.linalg.norm(x,axis=1,keepdims=True)
    x=x/x_norm
    return x,x_norm
x=np.array([[0, 3, 4],[1, 6, 4]])
print(normalize(x))