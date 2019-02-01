# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:23:46 2019

@author: Ahmed Alhag
"""

# XGBoost

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import datasets
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Churn_Modelling.csv';
print(FilePath)
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

# Fitting the XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

# Creating the prediction and Evaluation
Y_pred = classifier.predict(X_test)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying the k-fold cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs=-1)
ACC_mean = accuracies.mean()
ACC_std = accuracies.std()
