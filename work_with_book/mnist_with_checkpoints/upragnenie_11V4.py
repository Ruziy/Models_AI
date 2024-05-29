import numpy as np
import tensorflow as tf
import time

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Фильтрация данных для цифр 5-9
X_train_filtered = X_train[(y_train >= 5) & (y_train <= 9)] / 255.0
y_train_filtered = y_train[(y_train >= 5) & (y_train <= 9)] - 5
X_test_filtered = X_test[(y_test >= 5) & (y_test <= 9)] / 255.0
y_test_filtered = y_test[(y_test >= 5) & (y_test <= 9)] - 5

# Определение параметров
n_epochs = 100
batch_size = 20

# Выбор только 100 изображений на каждую цифру
num_images_per_digit = 100
X_train_selected = []
y_train_selected = []

for digit in range(5, 10):
    indices = np.where(y_train_filtered == digit - 5)[0][:num_images_per_digit]
    X_train_selected.extend(X_train_filtered[indices])
    y_train_selected.extend(y_train_filtered[indices])

# Преобразование в массив numpy
X_train_selected = np.array(X_train_selected)
y_train_selected = np.array(y_train_selected)

# Создание объекта Dataset для обучающих данных и кеширование
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_selected, y_train_selected))
train_dataset = train_dataset.cache().shuffle(len(X_train_selected)).batch(batch_size)

# Загрузка сохраненной модели
pretrained_model = tf.keras.models.load_model('нейронка_по_книге/mnist_with_checkpoints/V1_checkpoints.keras')

# Замораживаем все слои предварительно обученной модели
for layer in pretrained_model.layers:
    layer.trainable = False

# Создание новой DNN с использованием замороженных скрытых слоев предыдущей модели
new_model = tf.keras.Sequential([
    *pretrained_model.layers[:-1],  
    tf.keras.layers.Dense(5, activation='softmax', name='new_output_layer')  
])

# Компиляция новой модели
new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Начало замера времени
start_time = time.time()

new_model.fit(train_dataset, epochs=n_epochs,
          validation_data=(X_test_filtered, y_test_filtered), verbose=2)

# Завершение замера времени
end_time = time.time()

# Оценка производительности новой модели на тестовых данных
test_loss, test_accuracy = new_model.evaluate(X_test_filtered, y_test_filtered, verbose=0)
print("Final test accuracy: {:.2f}%".format(test_accuracy * 100))

# Вывод времени обучения
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))

#WithOut cache
#Final test accuracy: 80.21%
#Training time: 13.73 seconds
#With cache
#Final test accuracy: 80.35%
#Training time: 13.24 seconds