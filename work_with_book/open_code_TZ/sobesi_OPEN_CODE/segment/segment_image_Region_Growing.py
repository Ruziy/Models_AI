from skimage import io, segmentation, color
import matplotlib.pyplot as plt

# Загрузка изображения
image = io.imread('images\img_2_flow.jpg')
image_gray = color.rgb2gray(image)

# Применение алгоритма роста регионов
region_segmentation = segmentation.felzenszwalb(image, scale=100)

# Отображение результатов
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(region_segmentation, cmap='tab20')
plt.title('Region Growing Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
