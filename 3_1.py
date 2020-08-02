#------------------------------------Question-03_1--Wikasitha-------------------------------------

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

#adding noise to the data
noise_factor = 0.25

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0 , scale = 1.0, size = x_test.shape)

x_train_noisy = np.clip(x_train_noisy , 0. , 1.)
x_test_noisy = np.clip(x_test_noisy , 0. , 1.)

#scaling data 
x_train1 = normalize(x_train,axis=-1, order=2)
x_test1 = normalize(x_test,axis=-1, order=2)

#encoding target data using one hot encode method
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


#making the nueral network 
#two hidden layer with 16 neurons and 32 neurons
model = Sequential()
model.add(Dense(16, input_dim=784, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#Compilng the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#Fitting training data
model.fit(x_train1, y_train, epochs=10, batch_size=20)

#displaying accuracy 
accuracy = model.evaluate(x_test1, y_test)
print(accuracy)