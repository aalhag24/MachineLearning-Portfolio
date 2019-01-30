# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:16:13 2019

@author: Ahmed Alhag
"""

# Artificial Neural Networks

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.impute import SimpleImputer

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import datasets
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Churn_Modelling.csv';
dataset = pd.read_csv(FilePath)

# Aquire independent(X) and dependent(Y) columns
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0);

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating the ANN
from keras.models import Sequential
from keras.layers import Dense

# Initialzing the ANN
classifier = Sequential()

# Add the input and hidden layer 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Add the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the classifier (ANN) to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# Creating the prediction and Evaluation
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
acc = 1703/2000