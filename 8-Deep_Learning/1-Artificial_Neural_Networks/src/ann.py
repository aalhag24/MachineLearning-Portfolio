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

from pathlib import Path

cwd = Path().resolve()

# Check that the gpu/cpu is properly being used
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
def build_classifier(Optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = Optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Initialzing the ANN
classifier = Sequential()

# Add the input and hidden layer 
classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))

# Add the second hidden layer
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))

# Add the output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the classifier (ANN) to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# Creating the prediction and Evaluation
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Predicted a new test parameter(Observation)
new_pred = classifier.predict(sc_X.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
mean = accuracies.mean()
std = accuracies.std()

# Improving the ANN, Dropout Regularization to reduce overfitting
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
              'nb_epoch': [50, 100, 250, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_param = grid_search.best_params_
best_acc = grid_search.best_score_