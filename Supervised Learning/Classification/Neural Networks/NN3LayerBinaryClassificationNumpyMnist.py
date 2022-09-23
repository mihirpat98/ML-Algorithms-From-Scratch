import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_x_1 = []
train_y_1 = []
for i in range(len(train_X)):
    if train_y[i] == 1 or train_y[i] == 0:
        train_x_1.append(train_X[i].flatten())
        train_y_1.append(train_y[i])


def sigmoid(a):
    return 1/(1+np.exp(-a))


def my_dense(A_in, W, b, g):
    Z = np.matmul(A_in, W)+b
    a_out = sigmoid(Z)
    return(a_out)

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)