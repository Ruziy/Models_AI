import cv2
import numpy as np
import os
from tqdm import tqdm

# Загрузка модели YOLO v3
net = cv2.dnn.readNet("work_with_book\\open_code_TZ\\pipeline_live_detection\\yolov3.weights", "work_with_book\\open_code_TZ\\pipeline_live_detection\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка меток классов
with open("work_with_book\\open_code_TZ\\pipeline_live_detection\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Класс для фильтрации
target_class = "person"
target_class_id = classes.index(target_class)

# Функция для обработки изображения и сохранения предсказаний
def process_and_annotate_image(image_path, output_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Подготовка изображения для YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Обработка предсказаний YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == target_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Неперекрывающаяся разметка объектов
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    annotated_data = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == target_class:
                annotated_data.append([label, x, y, w, h])

    # Сохранение разметки в файл
    if annotated_data:
        with open(output_path, "w") as f:
            for data in annotated_data:
                f.write(" ".join(map(str, data)) + "\n")

# Путь к папке с изображениями
image_folder = "work_with_book\\open_code_TZ\\create_datasets\\images"
output_folder = "work_with_book\\open_code_TZ\\create_datasets\\annotations"

# Создание папки для разметки
os.makedirs(output_folder, exist_ok=True)

# Счетчик обработанных изображений
processed_count = 0

# Обработка всех изображений в папке с отображением прогресса
for image_file in tqdm(os.listdir(image_folder)):
    if processed_count >= 4000:
        break
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        process_and_annotate_image(image_path, output_path)
        processed_count += 1
 