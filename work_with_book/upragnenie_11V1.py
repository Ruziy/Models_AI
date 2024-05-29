import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input,BatchNormalization
import matplotlib.pyplot as plt
#Загрузка данных
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Нормализация по пиксилям
x_train = X_train.reshape(-1, 28*28)
x_test = X_test.reshape(-1, 28*28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Параметры модели
batch_size = 50
num_classes = 10
epochs = 40

initializer = tf.keras.initializers.HeNormal()
model = Sequential()
layer_1=model.add(Dense(300, use_bias=True, activation='elu', input_shape=(x_train.shape[1],)))
layer_2=model.add(Dense(100, use_bias=True, activation='elu',input_shape=layer_1 , kernel_initializer=initializer))
layer_3=model.add(Dense(100, use_bias=True, activation='elu',input_shape=layer_2 , kernel_initializer=initializer))
layer_4=model.add(Dense(100, use_bias=True, activation='elu',input_shape=layer_3 , kernel_initializer=initializer))
layer_5=model.add(Dense(100, use_bias=True, activation='elu',input_shape=layer_4 , kernel_initializer=initializer))
logis=model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

print(max(history.history['val_accuracy'])) ##0.9804999828338623
