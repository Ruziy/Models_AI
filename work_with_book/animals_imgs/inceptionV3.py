import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Загрузка и предобработка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape)
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0
print(x_test.shape)
x_train_resized = tf.image.resize(x_train, [75, 75])
x_test_resized = tf.image.resize(x_test, [75, 75])
print(x_test_resized.shape)
x_train_resized = tf.concat([x_train_resized] * 3, axis=-1)  # Преобразование в RGB формат
x_test_resized = tf.concat([x_test_resized] * 3, axis=-1)
print(x_test_resized.shape)
# Загрузка предварительно обученной модели InceptionV3 и заморозка весов
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
# for layer in base_model.layers:
#     layer.trainable = False

# # Добавление слоев классификации поверх базовой модели
# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(512, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

# # Компиляция и обучение модели
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train_resized, y_train, batch_size=32, epochs=5, validation_data=(x_test_resized, y_test))

# # Оценка производительности модели
# test_loss, test_acc = model.evaluate(x_test_resized, y_test)
# print("Test accuracy:", test_acc)