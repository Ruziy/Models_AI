import numpy as np
import tensorflow as tf

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормализация данных
X_train = X_train / 255.0
X_test = X_test / 255.0

# Преобразование меток в формат one-hot
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Создание модели DNN_A
DNN_A = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
])

# Создание модели DNN_B
DNN_B = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
])

# Объединение выходов обеих сетей DNN
concat_layer = tf.keras.layers.Concatenate(axis=1)
hidden_layer = tf.keras.layers.Dense(10, activation='elu', kernel_initializer='he_normal')
output_layer = tf.keras.layers.Dense(10, activation='softmax')  # Изменено на softmax

# Входные тензоры для обеих моделей
input_A = tf.keras.layers.Input(shape=(28, 28))
input_B = tf.keras.layers.Input(shape=(28, 28))

# Прогон данных через обе модели
output_A = DNN_A(input_A)
output_B = DNN_B(input_B)

# Объединение выходов обеих сетей DNN
combined_output = concat_layer([output_A, output_B])

# Пропуск объединенного выхода через скрытый слой
combined_output_hidden = hidden_layer(combined_output)

# Создание выходного слоя
output = output_layer(combined_output_hidden)

# Создание объединенной модели
combined_model = tf.keras.models.Model(inputs=[input_A, input_B], outputs=output)

# Компиляция модели
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Изменено на категориальную кросс-энтропию

# Создание случайных индексов для перемешивания
train_indices = list(range(len(X_train)))
test_indices = list(range(len(X_test)))
np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

# Перемешивание данных
X_train_shuffled = X_train[train_indices]
y_train_shuffled = y_train_one_hot[train_indices]
X_test_shuffled = X_test[test_indices]
y_test_shuffled = y_test_one_hot[test_indices]

# Обучение объединенной модели
combined_model.fit([X_train_shuffled, X_train_shuffled], y_train_shuffled, epochs=5, validation_data=([X_test_shuffled, X_test_shuffled], y_test_shuffled))

# Предсказание на тестовом наборе данных
y_pred = combined_model.predict([X_test_shuffled, X_test_shuffled])

# Применение порогового значения
threshold = 0.5
y_pred_binary = np.argmax(y_pred, axis=1)
count_ones = np.sum(y_pred_binary == 1)
# Вывод результата
print("Predictions:", len(y_pred_binary), "True:", count_ones)