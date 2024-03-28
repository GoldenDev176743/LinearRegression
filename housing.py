import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

housingData = pd.read_csv('housing.csv')

housingData.head()

x = housingData.iloc[:, :-1].values
y = housingData.iloc[:, [-1]].values

missingValueImputer = SimpleImputer()
x[:, :-1] = missingValueImputer.fit_transform(x[:, :-1])
y = missingValueImputer.fit_transform(y)

x_labelencoder = LabelEncoder()
x[:, -1] = x_labelencoder.fit_transform(x[:, -1])

X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)


knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
predictions = knn_regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error (MSE):', mse)