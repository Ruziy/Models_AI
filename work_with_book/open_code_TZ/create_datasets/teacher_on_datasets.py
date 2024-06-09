import tensorflow as tf
from tensorflow.keras import layers, models
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Функция для загрузки и предобработки датасета
def load_dataset(data_folder, target_class="person", max_annotations=10):
    image_paths = []
    annotations_list = []
    for annotation_file in os.listdir(data_folder):
        if annotation_file.endswith(".txt"):
            image_path = os.path.join("work_with_book\\open_code_TZ\\create_datasets\\images", os.path.splitext(annotation_file)[0] + ".jpg")
            annotations = []
            with open(os.path.join(data_folder, annotation_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_label = parts[0]
                    if class_label == target_class:
                        x_center, y_center, width, height = map(float, parts[1:])
                        annotations.append([x_center, y_center, width, height])
            if annotations:
                image_paths.append(image_path)
                # Паддинг или обрезка аннотаций до max_annotations
                annotations = annotations[:max_annotations]
                annotations += [[0, 0, 0, 0]] * (max_annotations - len(annotations))
                annotations_list.append(annotations)
    return image_paths, annotations_list

# Загрузка датасета
image_paths, annotations_list = load_dataset("work_with_book\\open_code_TZ\\create_datasets\\annotations")

# Разделение на обучающий и валидационный наборы
train_image_paths, val_image_paths, train_annotations, val_annotations = train_test_split(image_paths, annotations_list, test_size=0.2, random_state=42)

# Функция для предобработки изображений и аннотаций
def preprocess_image(image_path, annotations):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (416, 416)) / 255.0
    annotations = tf.reshape(tf.convert_to_tensor(annotations, dtype=tf.float32), [-1])
    return image, annotations

# Подготовка датасета
def create_tf_dataset(image_paths, annotations_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotations_list))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y))
    return dataset

train_dataset = create_tf_dataset(train_image_paths, train_annotations)
val_dataset = create_tf_dataset(val_image_paths, val_annotations)

# Определение модели CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(40)  # Вывод для 10 объектов: x_center, y_center, width, height для каждого
])

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Определение пути для сохранения модели с расширением .keras
model_save_path = "work_with_book/open_code_TZ/create_datasets/model_on_dataset_Yolov3.keras"

# Коллбэк для ранней остановки на основе валидационной точности
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Коллбэк для сохранения модели на основе валидационной точности
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max', verbose=2)

# Начало замера времени
start_time = time.time()

# Обучение модели с использованием коллбэков для ранней остановки и сохранения модели
model.fit(train_dataset.batch(32), 
          epochs=15, 
          validation_data=val_dataset.batch(32), 
          verbose=2, 
          callbacks=[checkpoint_callback, early_stopping_callback])

# Завершение замера времени
end_time = time.time()

# Оценка производительности модели на валидационных данных
test_loss, test_accuracy = model.evaluate(val_dataset.batch(32), verbose=0)
print("Final test accuracy: {:.2f}%".format(test_accuracy * 100))

# Вывод времени обучения
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))