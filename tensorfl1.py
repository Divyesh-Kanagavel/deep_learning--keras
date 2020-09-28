import tensorflow as tf
print(tf.__version__)
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()  # classic hand-written digit classification

import matplotlib.pyplot as plt

#plt.imshow(x_train[0],cmap = plt.cm.binary)  to check if we are getting the right inputs
#plt.show()

#print(y_train[0])

#Normalization

x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()

model = tf.keras.models.Sequential()   # A simple fully connected neural network

model.add(tf.keras.layers.Flatten())  # conveting 28*28 image into a 1*784 vector of values
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))   # hidden layer 1 - rectified linear unit
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  #hidden layer 2

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #output layer - softmax activation

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 4)

#at end of 4 epochs, we get a loss of 0.053 and an accuracy of 98.31% 

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_acc)
print(val_loss)

predictions = model.predict(x_test)
#to check if our model is right on first dataset.

print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap = plt.cm.binary)
plt.show()




