import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import time

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Проблема в том, что sklearn просит данные в формате np, что требует ещё 1 конвертацию данных.
# Это безбожно жрёт оперативку 32гб не хватает, как альтернатива была выбрана Keras-Tuner ищи проект teacherV2_Keras_Tuner.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

# Определение кастомного классификатора
class CustomCNNKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_conv_layers=2, num_filters=32, kernel_size=(3, 3), pool_size=(2, 2),
                 activation='relu', learning_rate=0.01, batch_size=32, epochs=5, 
                 initializer=tf.keras.initializers.HeNormal(), rate=0.2, 
                 use_batch_norm=False, l1_reg=0.0, l2_reg=0.0, padding='same'):
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.initializer = initializer
        self.rate = rate
        self.use_batch_norm = use_batch_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.padding = padding
        self.model = None  # Initialize model as None
        
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(self.num_filters, self.kernel_size, padding=self.padding,
                                kernel_initializer=self.initializer, 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg), 
                                input_shape=(416, 416, 3)))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(self.activation))
        model.add(layers.MaxPooling2D(self.pool_size))
        
        for _ in range(self.num_conv_layers - 1):
            model.add(layers.Conv2D(self.num_filters * 2, self.kernel_size, padding=self.padding,
                                    kernel_initializer=self.initializer, 
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)))
            if self.use_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(self.activation))
            model.add(layers.MaxPooling2D(self.pool_size))
            self.num_filters *= 2
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=self.activation, kernel_initializer=self.initializer,
                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)))
        if self.use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.rate))
        model.add(layers.Dense(4))  # Output layer for bounding box coordinates
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        if self.model is None:
            self.model = self._build_model()  # Build model if not already built
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Параметры для RandomizedSearchCV
param_distributions = {
    # 'num_conv_layers': [1, 2, 3],
    # 'num_filters': [32, 64, 128],
    # 'kernel_size': [(3, 3), (5, 5)],
    # 'pool_size': [(2, 2), (3, 3)],
    # 'activation': ['relu', 'elu'],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64],
    'epochs': [5, 10],
    # 'rate': [0.2, 0.3, 0.4],
    # 'use_batch_norm': [True, False],
    # 'l1_reg': [0.0, 0.01],
    # 'l2_reg': [0.0, 0.01],
    # 'padding': ['valid', 'same']
}

model = CustomCNNKerasClassifier()
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, verbose=3)

# Запуск RandomizedSearchCV на numpy массивах
random_search.fit(train_dataset.batch(32))

# Оценка производительности модели на валидационных данных
best_model = random_search.best_estimator_



# Укажите путь, по которому вы хотите сохранить файл
file_path = 'best_params.txt'
# Открытие файла для записи
with open(file_path, 'w') as f:
    # Запись параметров в файл
    for key, value in random_search.best_params_.items():
        f.write(f"{key}: {value}\n")

val_predictions = best_model.predict(val_dataset.batch(32))
val_accuracy = np.mean(np.all(np.isclose(val_predictions,val_dataset.batch(32), atol=0.1), axis=1))  # Оценка точности предсказания

print("Best parameters found: ", random_search.best_params_)
print("Validation accuracy: {:.2f}%".format(val_accuracy * 100))
