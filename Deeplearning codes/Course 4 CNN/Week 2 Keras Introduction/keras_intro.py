# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:10:08 2020

@author: Real
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def model(input_shape):
    # 定义一个占位符X_input，稍后人脸图片数据会输入到这个占位符中。input_shape中包含了占位符的维度信息. 
    X_input = Input(input_shape)

    # 给占位符矩阵的周边填充0
    X = ZeroPadding2D((3, 3))(X_input)

    # 构建一个卷积层，并对结果进行BatchNormalization操作，然后送入激活函数
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # 构建MAXPOOL层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # 将矩阵扁平化成向量，然后构建一个全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # 构建一个keras模型实例，后面会通过这个实例句柄来进行模型的训练和预测
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


def HappyModel(input_shape):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

happyModel = HappyModel(X_train.shape[1:])

print(X_train.shape[1:])
print(happyModel)


happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

happyModel.fit(X_train, Y_train, epochs=40, batch_size=50)


preds = happyModel.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


img_path = 'images/happy.jpg'

img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

print(img)
x = image.img_to_array(img)
# print(x)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))