import numpy as np
import copy
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
sgdr = SGDRegressor(max_iter=100000)
sgdr.fit(X_norm,y_train)

b_norm = sgdr.intercept_
w_norm = sgdr.coef_



## Make Predictions
y_pred_sgd = sgdr.predict(X_norm)
## OR
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(y_pred)