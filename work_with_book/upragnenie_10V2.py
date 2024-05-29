import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
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

model = Sequential()
model.add(Dense(300, use_bias=False, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(100, use_bias=False, activation='relu'))
model.add(Dense(num_classes, use_bias=False, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
print(max(history.history['val_accuracy']))

# Создать объект для подготовки данных
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=50).batch(50)


# Оценить обученную модель
# loss, accuracy = model.evaluate(x_test, y_test)

# Вывести результаты
# print(f"Точность: {accuracy}")

