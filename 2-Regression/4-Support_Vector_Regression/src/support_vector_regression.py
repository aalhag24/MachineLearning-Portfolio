# Support Vector Regression

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Import the data
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\\Position_Salaries.csv'))
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
dataset = pd.read_csv(Path)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

# Feature scaling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
#Y = pd.DataFrame(Y)

# Fitting a new result
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

# Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR result
SVR_Plot = plt.figure(1)
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth of Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
SVR_Plot.show()

# Visualising the Regression results (for higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
SVR_Plot2 = plt.figure(2)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff Salary Tester (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

SVR_Plot2.savefig(BinPath + '\SVR_PY.png')