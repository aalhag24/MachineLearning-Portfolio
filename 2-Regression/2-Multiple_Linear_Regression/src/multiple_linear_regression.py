# Multiple Linear Regression

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Import the data
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\\50_Startups.csv'))
dataset = pd.read_csv(Path)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Mulitple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the Test set
Y_pred = regressor.predict(X_test)

# Build an optimal model using Backward Elimination
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()


# Backward Elimination with p-values only
def backwardElimination(x, y, SL):
    numVar = len(x[0])
    for i in range(0,numVar):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if(maxVar > SL):
            for j in range(0, numVar - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_opt, Y, SL)


# Backward Elimination with p-values and Adjusted R squared
def backwardEliminationARS(x, y, SL):
    numVar = len(x[0])
    temp = np.zeros((50,6)).astype(float)
    for i in range(0, numVar):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVar-i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:,j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y,x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if(adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

X_ARS_modeled = backwardEliminationARS(X, Y, SL)
