import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
import numpy as np

# Загрузка и подготовка данных
(train_images, train_labels), (_, _) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0

# Создание модели для извлечения признаков
def create_feature_extraction_model(input_shape=(32, 32, 3)):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    return model

model = create_feature_extraction_model()

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Создание callback для ранней остановки
early_stopping = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)

# Извлечение признаков
features = model.predict(train_images)

# Применение K-means для кластеризации признаков
kmeans = KMeans(n_clusters=10, random_state=0).fit(features)
clusters = kmeans.labels_

# Оценка кластеров
accuracy = np.sum(clusters == train_labels.flatten()) / len(train_labels)
print(f'Clustering accuracy: {accuracy:.4f}')
