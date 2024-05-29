# import numpy as np
# import keras
# from tensorflow.keras import layers, models
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from tensorflow.keras.callbacks import ModelCheckpoint

# # Функция создания модели Keras
# def create_model(optimizer='adam', init='glorot_uniform', padding='valid', kernel_size=(3, 3), strides=(1, 1)):
#     model = models.Sequential([
#         layers.Conv2D(32, kernel_size=kernel_size, activation='relu', kernel_initializer=init, strides=strides, padding=padding, input_shape=(28, 28, 1)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, kernel_size=kernel_size, activation='relu', kernel_initializer=init, strides=strides, padding=padding),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, kernel_size=kernel_size, activation='relu', kernel_initializer=init, strides=strides, padding=padding),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu', kernel_initializer=init),
#         layers.Dense(10, activation='softmax')
#     ])
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Заворачивание модели Keras в KerasClassifier
# model = KerasClassifier(
#     model=create_model, 
#     verbose=2,
#     optimizer='adam',
#     init='glorot_uniform',
#     padding='valid',
#     kernel_size=(3, 3),
#     strides=(1, 1)
# )

# # Пространство поиска гиперпараметров
# param_dist = {
#     'optimizer': ['adam', 'rmsprop', 'sgd'],
#     # 'epochs': [10, 20, 30],
#     # 'batch_size': [32, 64, 128],
#     # 'init': ['glorot_uniform', 'normal', 'uniform'],
#     # 'padding': ['valid', 'same'],
#     # 'kernel_size': [(3, 3), (5, 5), (7, 7)],
#     # 'strides': [(1, 1), (2, 2)]
# }

# # Создание RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)

# # Загрузка данных MNIST
# from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# # # Добавление ModelCheckpoint callback для сохранения модели
# # checkpoint_callback = ModelCheckpoint(filepath='best_model_CNN.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# # # Включение callback в RandomizedSearchCV
# # callbacks = [checkpoint_callback]
# # model.callbacks = callbacks

# # Запуск RandomizedSearchCV
# random_search_result = random_search.fit(x_train, y_train)

# # Вывод лучших параметров
# print(f"Best: {random_search_result.best_score_} using {random_search_result.best_params_}")

# # Сохранение лучших параметров модели
# best_model = random_search_result.best_estimator_.model_
# best_model.save('нейронка_по_книге/animals_imgsbest_model_CNN_koef.h5')
#=====================================================================
import numpy as np
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import ModelCheckpoint

# Функция создания модели Keras с параметрами
def create_model(optimizer='adam', init='glorot_uniform', padding='valid', kernel_size=(3, 3), strides=(1, 1), num_layers=2, num_filters=32):
    model = models.Sequential()
    model.add(layers.Conv2D(num_filters, kernel_size=kernel_size, activation='relu', kernel_initializer=init, strides=strides, padding=padding, input_shape=(28, 28, 1)))
    
    for _ in range(num_layers):
        num_filters *= 2
        model.add(layers.Conv2D(num_filters, kernel_size=kernel_size, activation='relu', kernel_initializer=init, strides=strides, padding=padding))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# Заворачивание модели Keras в KerasClassifier
model = KerasClassifier(
    model=create_model, 
    verbose=2,
    optimizer='adam',
    init='glorot_uniform',
    padding='valid',
    kernel_size=(3, 3),
    strides=(1, 1),
    num_layers=2,
    num_filters=32,
    epochs=10,
    batch_size=32,
)

# Пространство поиска гиперпараметров
param_dist = {
    # 'optimizer': ['adam', 'rmsprop', 'sgd'],
    # 'epochs': [10, 20, 30],
    # 'batch_size': [32, 64, 128],
    # 'init': ['glorot_uniform', 'normal', 'uniform'],
    # 'padding': ['valid', 'same'],
    'kernel_size': [(3, 3), (5, 5), (7, 7)],
    'strides': [(1, 1), (2, 2)],
    'num_layers': [1, 2, 3],
    'num_filters': [16, 32, 64]
}

# Создание RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)

# Загрузка данных MNIST
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Запуск RandomizedSearchCV
random_search_result = random_search.fit(x_train, y_train)

# Вывод лучших параметров
print(f"Best: {random_search_result.best_score_} using {random_search_result.best_params_}")

# Сохранение лучших параметров модели
best_model = random_search_result.best_estimator_.model_
best_model.save('best_model_final.h5')