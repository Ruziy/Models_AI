import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
import matplotlib.pyplot as plt

# Генератор данных
def video_data_generator(num_samples, time_steps, height, width, channels, num_classes, batch_size):
    while True:
        X = np.random.rand(batch_size, time_steps, height, width, channels)
        y = np.random.randint(0, num_classes, batch_size)
        y = tf.keras.utils.to_categorical(y, num_classes)
        yield X, y


num_samples = 100
time_steps = 10
height = 64
width = 64
channels = 3
num_classes = 5
batch_size = 32


model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(time_steps, height, width, channels)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_generator = video_data_generator(num_samples, time_steps, height, width, channels, num_classes, batch_size)
history = model.fit(train_generator, steps_per_epoch=num_samples // batch_size, epochs=20, verbose=0)
X_test, y_test = next(video_data_generator(num_samples, time_steps, height, width, channels, num_classes, batch_size))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Loss: {loss}, Accuracy: {accuracy}')


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
