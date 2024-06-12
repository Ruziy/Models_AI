import cv2
import torch
import numpy as np
from torchvision import transforms

# Загрузка предобученной модели YOLO-TLA
# Замените 'path_to_yolotla_weights' на путь к файлу с весами модели
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_yolotla_weights')

def process_video(input_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео, выполняя детекцию людей и сохраняя результат.
    Считает количество распознанных людей за все видео.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model: Модель YOLO-TLA для детекции.
        conf_threshold (float): Порог уверенности для детекции.

    Returns:
        int: Общее количество распознанных людей.
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_people_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Преобразование кадра в формат, который ожидает модель YOLO-TLA
        results = model(frame)

        # Извлечение результатов обнаружения объектов
        labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        frame_people_count = 0
        for i in range(len(labels)):
            if labels[i] == 0 and coords[i, 4] >= conf_threshold:  # '0' предполагается как метка для человека
                x1, y1, x2, y2 = int(coords[i, 0] * width), int(coords[i, 1] * height), int(coords[i, 2] * width), int(coords[i, 3] * height)
                conf = coords[i, 4].item()
                
                # Отрисовка прямоугольника и текста на кадре
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'Person {conf:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                frame_people_count += 1
        
        total_people_count += frame_people_count
        out.write(frame)
    
    cap.release()
    out.release()
    
    return total_people_count

def main():
    """
    Основная точка входа в программу.
    """
    input_path = 'work_with_book/BBN_LLC/video/crowd.mp4'
    output_path = 'work_with_book/BBN_LLC/video/output_crowd.mp4'
    
    total_people_count = process_video(input_path, output_path, model, conf_threshold=0.25)
    print(f'Общее количество распознанных людей: {total_people_count}')

if __name__ == "__main__":
    main()
