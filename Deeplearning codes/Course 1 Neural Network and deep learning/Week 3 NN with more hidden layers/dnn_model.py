# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:19:47 2020

@author: Real
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import imageio
from os import listdir
from PIL import Image
from scipy import ndimage
from testCases_v3 import *
from dnn_app_utils_v3 import *


'''
from Initialize_Parameters import initialize_parameters_deep
from L_Model_Forward import L_model_forward
from L_Model_Backward import L_model_backward
from Update_Parameters import update_parameters
from Compute_Cost import compute_cost
from Predict import predict
'''
#matplotlib inline
start_time = time.time()


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)
#Load the data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
'''
#=====================================================
# #show an image in the dataset. Example of a picture
index = 12
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") + " picture.")
#=====================================================
# Explore your dataset
'''
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T #The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.0
test_x = test_x_flatten/255.0



# GRADED FUNCTION: L_layer_model

# GRADED FUNCTION: L_layer_model

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
# L_layer_model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)



#test image

loaded_images = list()
my_label_y = [1,1,1,0,0,0]
i=0
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
    my_image = my_image/255.
    print(my_image.shape)
    my_predicted_image = predict(my_image, my_label_y[i], parameters)
#plt.imshow(image)
    print("Filename:"+filename+" y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\"picture.")
    #img_org.show()
    loaded_images.append(img_org)
    i=i+1
loaded_images[-1].show()
####################################

#printing mislabeled images of test set below
print_mislabeled_images(classes, test_x, test_y, pred_test)
print("---Execution time  %s seconds ---" % (time.time() - start_time))