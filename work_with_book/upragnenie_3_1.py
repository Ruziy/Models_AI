from scipy.ndimage.interpolation import shift
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx])
    return shifted_image.reshape([-1])

def load_data(path="mnist.npz"):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data("нейронка_по_книге\mnist.npz")

x_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
x_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255

image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

# X_train_augmented = [image for image in x_train]
# y_train_augmented = [label for label in y_train]

# for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#     for image, label in zip(X_train, y_train):
#         X_train_augmented.append(shift_image(image, dx, dy))
#         y_train_augmented.append(label)

# X_train_augmented = np.array(X_train_augmented)
# y_train_augmented = np.array(y_train_augmented)
# knn_clf = KNeighborsClassifier(weights="distance",n_neighbors=5,n_jobs=4)
# knn_clf.fit(X_train_augmented, y_train_augmented)
# y_pred = knn_clf.predict(x_test)
# res = accuracy_score(y_test, y_pred)
# # res = cross_val_score(knn_clf, X_train_augmented, y_train_augmented, cv=3, scoring="accuracy")
# print(res)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()