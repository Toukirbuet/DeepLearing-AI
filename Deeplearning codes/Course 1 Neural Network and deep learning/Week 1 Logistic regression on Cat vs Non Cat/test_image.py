# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:30:37 2020

@author: Real
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from os import listdir
from PIL import Image
import sigmoid
from initialize import initialize_with_zeros
import propagate
from predict import predict
from optimize import optimize
from prepare_data import prepare_data
def test_image(folder_name,num_px,d,classes):
   loaded_images = list()
   for filename in listdir(folder_name):
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
        return loaded_images
        #loaded_images[4].show()