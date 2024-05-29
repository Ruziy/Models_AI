import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(path="mnist.npz"):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data("нейронка_по_книге\mnist.npz")
x_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
x_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255
# Создание и обучение KNN модели
kn_model = KNeighborsClassifier()
# kn_model.fit(x_train, y_train)
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
grd_model = GridSearchCV(kn_model,param_grid,cv=5,n_jobs=-1)
grd_model.fit(x_train,y_train)

# res = cross_val_score(kn_model, x_train, y_train, cv=3, scoring="accuracy")
print("Accur: ", grd_model.best_score_)
print("Best params: ", grd_model.get_params())
pkl_filename = "grdKN_model.pkl" 
with open(pkl_filename, 'wb') as file: 
    pickle.dump(grd_model, file) 