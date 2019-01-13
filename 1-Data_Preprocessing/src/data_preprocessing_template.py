# Data preprocessing

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import datasets
Path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin\Data.csv'))
print(Path)
dataset = pd.read_csv(Path)
print(dataset)

# Aquire independent(X) and dependent(Y) columns
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean');
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting into Training and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0);
sc_X = StandardScaler();

# Feature Scaling
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X)
print(Y)
print(X_train)
print(X_test)