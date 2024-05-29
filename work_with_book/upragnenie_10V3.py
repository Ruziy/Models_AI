import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from keras.datasets import mnist    

# Загрузить данные MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормализация по пикселям
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Преобразовать обучающий набор данных в массив с размерностью 2
X_train = X_train.reshape(-1, 28 * 28)

# Преобразовать модель в объект MLPClassifier
mlp_classifier = MLPClassifier()

# Параметры для RandomizedSearchCV
param_distributions = {
    'hidden_layer_sizes': [(100,), (100, 100), (300, 100), (100, 100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'batch_size': [32, 50, 64, 128]
}

# Создать объект RandomizedSearchCV
random_search = RandomizedSearchCV(mlp_classifier, param_distributions, n_iter=10)

# Запустить поиск
random_search.fit(X_train, y_train)

# Получить лучшие гиперпараметры
best_params = random_search.get_params()

# Вывести лучшие гиперпараметры
print(best_params)