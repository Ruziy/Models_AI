import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Создание модели для предсказания угла поворота изображения
def create_rotation_model(input_shape=(32, 32, 3)):
    base_model = tf.keras.applications.ResNet50(weights=None, input_shape=input_shape, include_top=False)
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(4, activation='softmax')  # 4 угла поворота (0°, 90°, 180°, 270°)
    ])
    return model

# Функция для поворота изображения
def rotate_image(image, angle):
    return tf.image.rot90(image, k=angle)

# Загрузка и подготовка данных
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0

# Создание данных с углами поворота
angles = np.random.randint(0, 4, train_images.shape[0])
rotated_images = np.array([rotate_image(img, angle).numpy() for img, angle in zip(train_images, angles)])

# Создание модели
model = create_rotation_model(input_shape=(32, 32, 3))

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Создание callback для ранней остановки
early_stopping = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)

# Обучение модели
model.fit(rotated_images, angles, epochs=100, batch_size=32, callbacks=[early_stopping])

