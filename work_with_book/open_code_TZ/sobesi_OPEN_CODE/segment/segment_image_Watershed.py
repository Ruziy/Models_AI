import numpy as np
import cv2
import matplotlib.pyplot as plt

# Загрузка изображения в цветовом формате
image = cv2.imread('images\img_2_flow.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Переводим в RGB для правильного отображения в matplotlib

# Применение фильтрации или других операций предобработки, если необходимо

# Подготовка маркеров для водораздела
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Получение маркеров для алгоритма водораздела
markers = cv2.connectedComponents(thresh)[1]

# Применение алгоритма водораздела
segmented_image = cv2.watershed(image, markers)

# Отображение результатов
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='tab20')
plt.title('Watershed Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
