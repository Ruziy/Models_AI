import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

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

# Определение модели для Keras Tuner
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=(hp.Int('kernel_height', min_value=3, max_value=5, step=2),
                     hp.Int('kernel_width', min_value=3, max_value=5, step=2)),
        padding=hp.Choice('padding', values=['same', 'valid']),
        activation=hp.Choice('activation', values=['relu', 'elu']),
        input_shape=(416, 416, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    if hp.Boolean('batch_norm'):
        model.add(layers.BatchNormalization())

    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Int('filters_' + str(i), min_value=32, max_value=128, step=32),
            kernel_size=(hp.Int('kernel_height_' + str(i), min_value=3, max_value=5, step=2),
                         hp.Int('kernel_width_' + str(i), min_value=3, max_value=5, step=2)),
            padding=hp.Choice('padding_' + str(i), values=['same', 'valid']),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'elu'])
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        if hp.Boolean('batch_norm_' + str(i)):
            model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        activation=hp.Choice('dense_activation', values=['relu', 'elu'])
    ))

    if hp.Boolean('dropout'):
        model.add(layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(layers.Dense(40))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    return model

# Использование Keras Tuner для поиска гиперпараметров
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='work_with_book\open_code_TZ\create_datasets',
    project_name='keras_tuner'
)

# Обучение с использованием Keras Tuner
tuner.search(
    train_dataset.batch(32),
    validation_data=val_dataset.batch(32),
    epochs=kt.HyperParameters().Int('epochs', min_value=5, max_value=20, step=5)
)

# Получение лучших гиперпараметров
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Сохранение лучших гиперпараметров в файл
best_params_path = 'best_params.txt'
with open(best_params_path, 'w') as f:
    for param, value in best_hyperparameters.values.items():
        f.write(f"{param}: {value}\n")

print("Best hyperparameters saved to:", best_params_path)