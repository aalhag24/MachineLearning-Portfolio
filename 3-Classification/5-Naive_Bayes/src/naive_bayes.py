# Naive Bayesian Classification

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB

# Import the data
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\\Social_Network_Ads.csv'))
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
dataset = pd.read_csv(Path)

X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# Splitting into Training and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting a new result
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Create the Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualization the training set
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

TrainingPlot = plt.figure(1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('darkred', 'darkgreen'))(i), 
                label = j)
plt.title('Naive Bayesian Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
TrainingPlot.show()
TrainingPlot.savefig(BinPath + '/NaiveBases_Training_PY.png')


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
plt.title('Naive Bayesian Classification (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

TestingPlot.savefig(BinPath + '/NaiveBases_Testing_PY.png')