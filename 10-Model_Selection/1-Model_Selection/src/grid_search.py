# -*- coding: utf-8 -*-
"""
Created on Fri Feb 01 12:40:13 2019

@author: Ahmed Alhag
"""

# Grid Search

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Social_Network_Ads.csv'
dataset = pd.read_csv(FilePath)

X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# Splitting into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting a new result
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',
                 random_state = 0)
classifier.fit(X_train, Y_train);

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Create the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying the k-fold cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs=-1)
ACC_mean = accuracies.mean()
ACC_std = accuracies.std()

# Applying Grid Search to find the best model and the best parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8]} ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Applying the Best parameters
classifier = SVC(kernel = best_parameters['kernel'],
                 gamma = best_parameters['gamma'],
                 random_state = 0)
classifier.fit(X_train, Y_train);
Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

# Visualization the training set
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

from matplotlib.colors import ListedColormap
TrainingPlot = plt.figure(1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('darkred', 'darkgreen'))(i), 
                label = j)
plt.title('GridSearch - Kernal Support Vector Machine (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
TrainingPlot.show()
TrainingPlot.savefig(BinPath + '\GridSearch_Training_PY.png')


# Visualization the test set
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

TestingPlot = plt.figure(2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('darkred', 'darkgreen'))(i), 
                label = j)
plt.title('GridSearch - Kernal Support Vector Machine (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
TestingPlot.savefig(BinPath + '\GridSearch_Testing_PY.png')