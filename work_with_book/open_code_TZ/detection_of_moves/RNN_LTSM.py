import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Загрузка предварительно обученной модели
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(30, 3780)))  # 30 кадров, 3780 признаков на кадр
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Функция для извлечения признаков
def extract_features(frame):
    # Пример: использование HOG для извлечения признаков
    hog = cv2.HOGDescriptor()
    h = hog.compute(frame)
    return h.flatten()

# Функция для распознавания действий
def recognize_action(frames):
    features = [extract_features(frame) for frame in frames]
    features = np.array(features).reshape(1, 30, -1)
    prediction = model.predict(features)
    return prediction

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)
frames = []
while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (64, 128))  # Убедитесь, что кадры имеют фиксированный размер для HOG
    frames.append(frame)
    if len(frames) == 30:  # Анализировать каждые 30 кадров
        action = recognize_action(frames)
        print("Action:", action)
        frames = []
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()
