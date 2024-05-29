import numpy as np
import tensorflow as tf

# Создание кодировщика
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu')
])

# Создание декодировщика
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(28*28, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])

# Объединение кодировщика и декодировщика в автоэнкодер
autoencoder = tf.keras.models.Sequential([
    encoder,
    decoder
])

# Компиляция модели
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Загрузка данных MNIST
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Нормализация данных
X_train = X_train / 255.0
X_test = X_test / 255.0

# Обучение модели
autoencoder.fit(X_train, X_train, epochs=10, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

# Предсказание на тестовом наборе данных
decoded_images = autoencoder.predict(X_test)

# Визуализация входных и восстановленных изображений
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Входное изображение
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Восстановленное изображение
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()