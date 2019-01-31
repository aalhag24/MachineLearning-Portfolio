# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:07:32 2019

@author: Ahmed Alhag
"""

# Convolutional Neural Network

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import Keras libraries Material
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import datasets
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
TrainingData = BinPath + '/dataset/training_set';
TestingData = BinPath + '/dataset/test_set';

# Initialising the CNN
classifier = Sequential()

# Setting up the CNN Classifier
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling/Downsampling the data
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Second Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten the model into a sequentail data form
classifier.add(Flatten())

# Create the Layers
classifier.add(Dense(output_dim = 128, 
                     activation = 'relu'))
classifier.add(Dense(output_dim = 1, 
                     activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Fitting the training and testing set
training_set = train_datagen.flow_from_directory(TrainingData,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory(TestingData,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fitting the CNN classifier to the data
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)