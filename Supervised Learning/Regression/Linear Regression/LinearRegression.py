import math, copy
import numpy as np


# Load our data set
x_train = np.array([1.0, 2.0,5.0,9.0])   #features
y_train = np.array([300.0, 500.0,900.0,1500.0])   #target value


def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost +(f_wb-y[i])**2
    total_cost = 1/(2*m)*cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] +b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i =  (f_wb - y[i])
        dj_db+= dj_db_i
        dj_dw+= dj_dw_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return    dj_dw, dj_db 

def gradient_descent(x,y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    b = b_in
    w = w_in 
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    return w,b

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(w_final,b_final)


import matplotlib.pyplot as plt
x = np.linspace(-5,10,1000)
y = w_final*x+b_final
plt.plot(x, y, '-r', label='Linear Regression')
plt.scatter(x_train,y_train,marker ='x')
plt.title('Graph of y=2x+1')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()