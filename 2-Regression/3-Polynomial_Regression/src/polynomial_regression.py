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

# Splitting the dataset
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the data
lin_reg = LinearRegression()
lin_reg.fit(X, Y);

# Fitting Polnomial Regressoin to the data
poly_reg_deg_2 = PolynomialFeatures(degree=2)
poly_reg_deg_3 = PolynomialFeatures(degree=3)
poly_reg_deg_4 = PolynomialFeatures(degree=4)

X_poly_deg_2 = poly_reg_deg_2.fit_transform(X)
X_poly_deg_3 = poly_reg_deg_3.fit_transform(X)
X_poly_deg_4 = poly_reg_deg_4.fit_transform(X)

lin_reg_deg_2 = LinearRegression()
lin_reg_deg_3 = LinearRegression()
lin_reg_deg_4 = LinearRegression()

lin_reg_deg_2.fit(X_poly_deg_2, Y)
lin_reg_deg_3.fit(X_poly_deg_3, Y)
lin_reg_deg_4.fit(X_poly_deg_4, Y)

# Visualising the Linear Regression results
LinearPlot = plt.figure(1);
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff Salary Tester (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
LinearPlot.show()

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
PolyPlot_deg_2 = plt.figure(2);
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_deg_2.predict(poly_reg_deg_2.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff Salary Tester (Degree 2 Poly)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
PolyPlot_deg_2.show()

PolyPlot_deg_3 = plt.figure(3);
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_deg_3.predict(poly_reg_deg_3.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff Salary Tester (Degree 3 Poly)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
PolyPlot_deg_3.show()

PolyPlot_deg_4 = plt.figure(4);
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_deg_4.predict(poly_reg_deg_4.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff Salary Tester (Degree 4 Poly)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_deg_2.predict(poly_reg_deg_2.fit_transform([[6.5]])))

print(lin_reg_deg_3.predict(poly_reg_deg_3.fit_transform([[6.5]])))

print(lin_reg_deg_4.predict(poly_reg_deg_4.fit_transform([[6.5]])))