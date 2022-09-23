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


X = np.array(train_x_1[:1000])
y = np.array(train_y_1[:1000])
X_test = np.array(train_x_1[1000:1300])
y_test = np.array(train_y_1[1000:1300])

model = Sequential(
    [               
        tf.keras.Input(shape=(784,)),    #specify input size
        ### START CODE HERE ### 
        Dense(25, activation='sigmoid', name = 'layer1' ),
        Dense(15, activation='sigmoid', name = 'layer2' ),
        Dense(1, activation='sigmoid', name = 'layer3' )
        ### END CODE HERE ### 
    ], name = "my_model" 
)                            

model.summary()
[layer1, layer2, layer3] = model.layers

#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")



model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.001),)
model.fit( X,y, epochs=20)



predictions = model.predict(X_test)
yhat = (predictions >= 0.5).astype(int)
count = 0
for i in range(len(yhat)):
    if yhat[i] == y_test[i]:
        count+=1
    else:
        pass

print(yhat)
print("Accuracy of the model is:" + str(count/len(yhat)*100))