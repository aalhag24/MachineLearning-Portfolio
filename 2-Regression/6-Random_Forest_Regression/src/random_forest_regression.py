# Random Forest Regression

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Import the data
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\\Position_Salaries.csv'))
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
dataset = pd.read_csv(Path)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

# Fitting a new result
regressor = RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(X,Y)

# Predicting a new result
Y_pred = regressor.predict(np.array([[6.5]]))
print(Y_pred)

# Visualising the SVR result
RFR = plt.figure(1)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter([6.5], Y_pred, color = 'green')
plt.title('Truth of Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

RFR.savefig(BinPath + '\RandomForestRegression_1000_Trees_PY.png')