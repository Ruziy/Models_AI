import tensorflow as tf

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Получение списка переменных подлежащих обучению
trainable_vars = model.trainable_variables

# Вывод списка переменных
for var in trainable_vars:
    print(var.name)