# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:47:44 2020

@author: Real
"""

import scipy
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot  as plt
from os import listdir
# import scipy.misc
# from scipy.misc import imresize

#my_image = "thumbs_up.jpg"
loaded_images = list()
Fname_images=list()
for filename in listdir('images'):

    fname = "images/" + filename
    Fname_images.append(fname)
    image = np.array(plt.imread(fname))
    # my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    # AttributeError: module 'scipy.misc' has no attribute 'imresize'  上边的方法会报错，所以使用下边的方法代替
    my_image = resize(image, output_shape=(64, 64)).reshape((1, 64 * 64 * 3)).T
    my_image_prediction = predict(my_image, parameters)
    #my_image_prediction =1
    
    
    
    #plt.imshow(image)
    print("Filename:"+filename+"Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
    loaded_images.append(image)
    
plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
num_images = len(loaded_images)
for i in range(num_images):
    
        
    plt.subplot(2, num_images, i + 1)
    plt.imshow(loaded_images[i])
    plt.axis('off')
    plt.title(Fname_images[i])

