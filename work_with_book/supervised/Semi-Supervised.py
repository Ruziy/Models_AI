import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Загрузка и подготовка данных
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Создание поднабора размеченных данных
labeled_indices = np.random.choice(len(train_images), 1000, replace=False)
unlabeled_indices = np.setdiff1d(np.arange(len(train_images)), labeled_indices)

labeled_images = train_images[labeled_indices]
labeled_labels = train_labels[labeled_indices]
unlabeled_images = train_images[unlabeled_indices]

# Создание модели
def create_classification_model(input_shape=(32, 32, 3)):
    base_model = tf.keras.applications.ResNet50(weights=None, input_shape=input_shape, include_top=False)
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')  # 10 классов CIFAR-10
    ])
    return model

model = create_classification_model()

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Создание callback для ранней остановки
early_stopping = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)

# Обучение модели на размеченных данных
model.fit(labeled_images, labeled_labels, epochs=100, batch_size=32, callbacks=[early_stopping])

# Генерация псевдо-меток для неразмеченных данных
pseudo_labels = model.predict(unlabeled_images).argmax(axis=1)

# Обучение модели на псевдо-размеченных данных
model.fit(unlabeled_images, pseudo_labels, epochs=100, batch_size=32, callbacks=[early_stopping])
