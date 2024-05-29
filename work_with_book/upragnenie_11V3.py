import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_hidden_layers=1, num_neurons=100, activation='relu', learning_rate=0.01, batch_size=32, epochs=5, initializer=tf.keras.initializers.HeNormal(), rate = 0.2):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.initializer = initializer
        self.rate = rate
        self.model = self._build_model()
        

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        for _ in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons, kernel_initializer=self.initializer))
            model.add(tf.keras.layers.BatchNormalization())  
            model.add(tf.keras.layers.Activation(self.activation))  
            model.add(tf.keras.layers.Dropout(self.rate))  
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

# Load MNIST data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define parameter space
param_dist = {
    "num_hidden_layers": [1, 2, 3],
    "num_neurons": [64, 128, 256],
    "activation": ['relu', 'elu'],
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "epochs": [5, 10, 20],
    "rate":[0.1,0.2,0.5]
}

# Run RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=CustomKerasClassifier(), param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)

# Evaluate the best parameters
print("Best parameters found: ", random_search.best_params_)

# Evaluate the performance of the best model on the test dataset
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy with best parameters: {:.2f}%".format(accuracy * 100))

#1 without initializer he,regulirizer,batch_normalization
#Test accuracy with best parameters: 97.29%
#Best parameters found:  {'num_neurons': 128, 'num_hidden_layers': 2, 'learning_rate': 0.1, 'epochs': 20, 'batch_size': 128, 'activation': 'elu'}

#2 with initializer he,regulirizer without batch_normalization
#Test accuracy with best parameters: 97.33%
#Best parameters found:  {'num_neurons': 256, 'num_hidden_layers': 1, 'learning_rate': 0.01, 'epochs': 20, 'batch_size': 128, 'activation': 'relu'}

#3 with initializer he,regulirizer,batch_normalization
#Test accuracy with best parameters: 97.85%
#Best parameters found:  {'num_neurons': 128, 'num_hidden_layers': 3, 'learning_rate': 0.01, 'epochs': 20, 'batch_size': 32, 'activation': 'relu'}

#4 with initializer he,regulirizer,batch_normalization + rate for dropout
#Test accuracy with best parameters: 97.92%