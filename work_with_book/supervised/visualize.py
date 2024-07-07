import tensorflow as tf
import matplotlib.pyplot as plt

# Загрузка данных CIFAR-10
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

# Метки классов CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Функция для отображения изображений
def plot_images(images, labels, class_names, rows=1, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[labels[i][0]])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Отобразим первые три изображения
plot_images(train_images[:3], train_labels[:3], class_names)
import tensorflow as tf

# Функция для переворота изображения на указанный угол
def rotate_image(image, angle):
    return tf.image.rot90(image, k=angle)

# Пример использования функции
angle = 1  # Угол поворота (0, 1, 2 или 3)
rotated_image = rotate_image(train_images[0], angle)

# Отображение исходного и перевернутого изображения
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(train_images[0])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Rotated Image ({angle * 90}°)')
plt.imshow(rotated_image)
plt.axis('off')

plt.tight_layout()
plt.show()
