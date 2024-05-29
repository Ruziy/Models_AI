import tensorflow as tf
from tensorflow.keras import layers, models


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",metrics = ["accuracy"],loss= 'sparse_categorical_crossentropy')
model.fit(x_train,y_train,epochs=5,validation_data=(x_test, y_test))
test_loss,test_acc = model.evaluate(x_test,y_test,verbose=2)
print(f'\nTest accuracy: {test_acc}')

#Test accuracy: 0.9889000058174133 without padding 
#Test accuracy: 0.9850999712944031 with padding