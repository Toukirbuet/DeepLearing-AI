# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:20:20 2020

@author: Real
"""
import numpy as np
def image2vector(image):

### START CODE HERE ### ( 1 line of code)
    ###print("height:",image[0][0][0])
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
### END CODE HERE ###
    return v
x=np.array([[[1,2],[1,3]],[[1,0],[0,1]]])
print(image2vector(x))