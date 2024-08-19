import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Загрузка данных
data = load_iris()
X, y = data.data, data.target

# Определение модели
model = RandomForestClassifier(random_state=42)

# Определение пространства гиперпараметров
param_space = {
    'n_estimators': Integer(10, 100),
    'max_depth': Integer(1, 10),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

# Байесовская оптимизация с использованием перекрестной проверки
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=32,  # Количество итераций
    scoring='accuracy',
    cv=5,  # Количество фолдов в кросс-валидации
    random_state=42
)

# Запуск оптимизации
bayes_search.fit(X, y)

# Вывод лучших гиперпараметров и метрики
print(f"Лучшие гиперпараметры: {bayes_search.best_params_}")
print(f"Лучшая метрика: {bayes_search.best_score_}")
