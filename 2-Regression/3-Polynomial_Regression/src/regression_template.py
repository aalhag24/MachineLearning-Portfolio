# Polynomial Linear Regression

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Import the data
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\\Position_Salaries.csv'))
dataset = pd.read_csv(Path)

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting Regressoin to the data
### Create the regressor here

# Predicting a new result with Non-Linear Regression
Y_pred_ = regressor.predict([[6.5]])

# Visualising the Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.plot(6.5, Y_pred, color = 'green')
plt.title('Truth or Bluff Salary Tester (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution)
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.plot(6.5, Y_pred, color = 'green')
plt.title('Truth or Bluff Salary Tester (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
