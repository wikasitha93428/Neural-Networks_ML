#------------------------------------Question-01--Wikasitha-------------------------------------

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import normalize
from keras.metrics import Precision

#loading  Minst data 
#splitting datai into train and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#converting 2 dimension data in numpy ndarray 
x_train = x_train.reshape(-1, 784).astype("float32")
x_test = x_test.reshape(-1, 784).astype("float32")

#scaling data 
x_train1 = normalize(x_train,axis=-1, order=2)
x_test1 = normalize(x_test,axis=-1, order=2)

#encoding target data using one hot encode method
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#making the nueral network 
#one hidden layer with 16 neurons 
model = Sequential()
model.add(Dense(16, input_dim=784, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#Compilng the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting training data
model.fit(x_train1, y_train, epochs=10, batch_size=15)

#displaying accuracy 
accuracy = model.evaluate(x_test1, y_test)
print(accuracy)