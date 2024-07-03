import cv2
import numpy as np

import matplotlib.pyplot as plt

# Функция для сегментации изображения с помощью K-means
def segment_image_kmeans(image_path, k=3):
    # Чтение изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Преобразование изображения в 2D массив
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Определение критериев и применение K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Преобразование центров в uint8
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    # Преобразование всех пикселей в цвет центров кластеров
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

# Путь к изображению (замените на путь к вашему изображению из датасета)
image_path = 'images\img_2_flow.jpg'

# Применение сегментации
segmented_image = segment_image_kmeans(image_path, k=3)

# Отображение оригинального и сегментированного изображений
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Оригинальное изображение')
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Сегментированное изображение')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
