import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
#Функция вывода для cross_val_score
def display_scores ( scores ): 
    print("Суммы оценок : " , scores ) 
    print( "Cpeднee : " , scores . mean ( ) ) 
    print( "Стандартное отклонение : " , scores . std ( ) ) 

df = pd.read_csv("нейронка_по_книге\handson-ml-master\datasets\housing\housing.csv")

housing_num = df.drop('ocean_proximity', axis=1)
housing_cat = df[['ocean_proximity']]
#Разбиение датасеты
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#Получение названия таблиц числовых и категориальных
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Создание копии данных для изучения без целевого столбца
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# Получение списка числовых и категориальных столбцов
num_attribs = list(housing.drop('ocean_proximity', axis=1))
cat_attribs = ["ocean_proximity"]

# Создание предобработочного конвейера
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Применение преобразований к данным
housing_prepared = full_pipeline.fit_transform(housing)

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }
svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
scores = cross_val_score(svm_reg, housing_prepared,housing_labels , scoring="neg_mean_squared_error",cv=10) 
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
print(rnd_search.best_estimator_)
