import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('images\img_2_flow.jpg', 0)  # Чтение в градациях серого

# Применение алгоритма Canny для обнаружения границ
edges = cv2.Canny(image, 100, 200)  # Подбор параметров можно изменить

# Отображение результатов
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge-based Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
