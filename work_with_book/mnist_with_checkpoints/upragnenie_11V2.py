import numpy as np
import tensorflow as tf

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормализация данных
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train = X_train[y_train<5]
y_train = y_train[y_train<5]
X_test = X_test[y_test<5]
y_test = y_test[y_test<5]
# Определение параметров
n_epochs = 1000
batch_size = 20
max_checks_without_progress = 20
initializer = tf.keras.initializers.HeNormal()

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Создание объекта Dataset для обучающих данных
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)

# Создание колбэка для ранней остановки
early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=max_checks_without_progress,
                                                           restore_best_weights=True)

# Создание колбэка для сохранения лучших параметров модели
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='нейронка_по_книге\mnist_with_checkpoints\V1_checkpoints.keras',
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         verbose=1)

# Обучение модели
model.fit(train_dataset, epochs=n_epochs, callbacks=[early_stopping_callback, checkpoint_callback],
          validation_data=(X_test, y_test), verbose=2)

# Оценка производительности модели на тестовых данных
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Final test accuracy: {:.2f}%".format(test_accuracy * 100))