# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:52:39 2020

@author: Real
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import sigmoid
from initialize import initialize_with_zeros
import propagate
from predict import predict
from optimize import optimize
my_image = "test2.jpg" # change this to the name of your image file
# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\"picture.")
