# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:48:25 2020

@author: Real
"""

#TF GPU CHECK

# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow as tf 
print('tensorflow: %s' % tf.__version__)
# keras
import keras
print('keras: %s' % keras.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# import tensorflow as tf 

# if tf.test.gpu_device_name(): 

#     print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

# else:

#    print("Please install GPU version of TF")

# import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
print(tf.test.gpu_device_name())