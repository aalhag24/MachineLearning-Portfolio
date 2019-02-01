# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:18:33 2019

@author: Ahmed Alhag
"""

# Linear Discriminant Analysis

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
FilePath = BinPath + '\wine.csv';
dataset = pd.read_csv(FilePath)

# Aquire independent(X) and dependent(Y) columns
X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:, 13].values

# Splitting into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0);

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Creating the PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Creating the prediction and Evaluation
Y_pred = classifier.predict(X_test)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualization the training set
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

LinearDiscriminantAnalysisTraining = plt.figure(1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green', 'blue'))(i), 
                label = j)
plt.title('Linear Discriminant Analysis (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
LinearDiscriminantAnalysisTraining.show()
LinearDiscriminantAnalysisTraining.savefig(BinPath + '\LDA_Training_PY.png')


# Visualization the test set
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

LinearDiscriminantAnalysisTesting = plt.figure(2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green', 'blue'))(i), 
                label = j)
plt.title('Linear Discriminant Analysis (Testing set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
LinearDiscriminantAnalysisTesting.savefig(BinPath + '\LDA_Testing_PY.png')