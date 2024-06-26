from skimage import io, color, filters
import matplotlib.pyplot as plt

# Загрузка и предобработка изображения
image = io.imread('images\img_1_flow.jpg')
image_gray = color.rgb2gray(image)

# Применение фильтра порога
threshold = filters.threshold_otsu(image_gray)  # Применение автоматического порога
binary = image_gray > threshold

# Отображение результатов
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap='gray')
plt.title('Threshold-based Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
