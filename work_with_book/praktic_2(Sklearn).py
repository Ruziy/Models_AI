import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
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
from sklearn.ensemble import RandomForestRegressor

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

#Демонстрация работы GridSearchCV 
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
scores = cross_val_score(forest_reg, housing_prepared,housing_labels , scoring="neg_mean_squared_error",cv=10) 
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
print(grid_search.best_estimator_)

#Обучение DecisionTreeRegressor
# model_tree = DecisionTreeRegressor()
# model_tree.fit(housing_prepared, housing_labels)
# scores = cross_val_score(model_tree, housing_prepared,housing_labels , scoring="neg_mean_squared_error",cv=10) 
# tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores)
#Сохранение модели
# pkl_filename = "tree_prak_2.pkl" 
# with open(pkl_filename, 'wb') as file: 
#     pickle.dump(model_tree, file) 