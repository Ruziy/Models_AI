from skimage import io, color, measure
import matplotlib.pyplot as plt

# Загрузка и предобработка изображения
image = io.imread('images\img_2_flow.jpg')
image_gray = color.rgb2gray(image)

# Применение алгоритма связных компонент
labels = measure.label(image_gray > 0.6)  # Пример использования порога для выделения объектов

# Отображение результатов
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(labels, cmap='nipy_spectral')
plt.title('Connected Component-based Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()

