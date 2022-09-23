import numpy as np
import matplotlib.pyplot as plt
import matplotlib

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])           

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_train)

b_out = lr_model.intercept_
w_out = lr_model.coef_





x1= []
x2 = []
colors = ['red','green']
for i in range(len(X_train)):
  x1.append(X_train[i][0])
  x2.append(X_train[i][1])
plt.scatter(x1,x2,c= y_train,cmap=matplotlib.colors.ListedColormap(colors),marker ='x')

x0 = -b_out/w_out[0][0]
x1 = -b_out/w_out[0][1]
plt.plot([0,x0],[x1,0], lw=1)
plt.title('Logistic Regression')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()