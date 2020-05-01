# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:02:36 2020

@author: Real
"""
### START CODE HERE ###
# initialize parameters with zeros ( 1 line of code)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import imageio
from os import listdir
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import sigmoid
from initialize import initialize_with_zeros
import propagate
from predict import predict
from optimize import optimize
from prepare_data import prepare_data
from test_image import test_image

train_set_x, train_set_y, test_set_x, test_set_y,num_px,classes=prepare_data()

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent ( 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples ( 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    ### END CODE HERE ###
    # Print train/test Errors
    print("train accuracy: {} %".format(100 -np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
    "Y_prediction_test": Y_prediction_test,
    "Y_prediction_train" : Y_prediction_train,
    "w" : w,
    "b" : b,
    "learning_rate" : learning_rate,
    "num_iterations": num_iterations}
    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
#loaded_images=test_image('images',num_px,d,classes)
#loaded_images[1].show() 
loaded_images = list()
for filename in listdir('images'):
	# load image
	#img_data = image.imread('images/' + filename)
	# store loaded image
	
	#print('> loaded %s %s' % (filename, img_data.shape))
    fname = "images/" + filename
    
    img_org=Image.open(fname)
    img = img_org.resize((num_px,num_px), Image.ANTIALIAS)
    #loaded_images.append(img_org)
    data=np.array(img)
    my_image = data.reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
#plt.imshow(image)
    print("Filename:"+filename+" y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\"picture.")
    #img_org.show()
    loaded_images.append(img_org)
loaded_images[-1].show()
    