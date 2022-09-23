import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()



X_1 = []
y_1 = []
for i in range(len(train_X)):
    X_1.append(train_X[i].flatten())
    y_1.append(train_y[i])
print(len(X_1))
X = np.array(X_1[:5700])
y = np.array(y_1[:5700])
X_test = np.array(X_1[5700:])
y_test = np.array(y_1[5700:])
'''****IMP***The input dimension of the first layer is derived from the size of the input data specified in the model.fit statement below.
Note: It is also possible to add an input layer that specifies the input dimension of the first layer. For example:
tf.keras.Input(shape=(400,)),    #specify input shape
We will include that here to illuminate some model sizing.'''

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [   tf.keras.Input(shape=(784,)),                   # FOR REGULARIZATION
        Dense(25, activation = "relu", name= "L1"), # Dense(units = 25, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.1)),                                                           
        Dense(15, activation = "relu", name= "L2"), # Dense(units = 15, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        Dense(10, activation = "linear", name= "L3")
    ], name = "my_model" 
)
model.build()
model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),)

model.fit(X,y,epochs=40)
predictions = model.predict(X_test)

prediction_p = tf.nn.softmax(predictions)
yhat = []
for i in range(len(prediction_p)):
    yhat.append(np.argmax(prediction_p[i]))
count = 0
for i in range(len(yhat)):
    if yhat[i] == y_test[i]:
        count+=1
    else:
        pass
print("Accuracy of the model is:" + str(count/len(yhat)*100))