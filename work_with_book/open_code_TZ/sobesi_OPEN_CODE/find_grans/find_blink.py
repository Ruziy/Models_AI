import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread(r'work_with_book\open_code_TZ\sobesi_OPEN_CODE\find_grans\imgs\img_1.jpeg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(800,600))
# Применение оператора Собеля
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Горизонтальные границы
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Вертикальные границы

# Преобразование результатов в 8-битные изображения
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Объединение горизонтальных и вертикальных границ
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Применение оператора Лапласа
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Применение оператора Превитта
# Превитт горизонтальные и вертикальные границы
prewitt_x = cv2.filter2D(image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
prewitt_y = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))

# Преобразование результатов в 8-битные изображения
prewitt_x = cv2.convertScaleAbs(prewitt_x)
prewitt_y = cv2.convertScaleAbs(prewitt_y)

# Объединение горизонтальных и вертикальных границ Превитта
prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# Применение оператора Кэнни
canny_edges = cv2.Canny(image, 100, 200)

# Показ результатов
cv2.imshow('Original', image)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Prewitt Combined', prewitt_combined)
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
