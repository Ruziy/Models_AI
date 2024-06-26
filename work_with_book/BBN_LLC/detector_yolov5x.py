import torch
import cv2
import numpy as np

def load_model():
    """
    Загружает предобученную модель YOLOv5x для детекции людей.
    Возвращает модель.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    model.classes = [0]  # Фильтрация только людей (класс 0 в COCO)
    return model

def process_video(input_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео, выполняя детекцию людей и сохраняя результат.
    Считает количество распознанных людей за все видео.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model: Модель YOLOv5 для детекции.
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
        
        results = model(frame)
        labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        
        frame_people_count = 0
        for label, coord in zip(labels, coords):
            if int(label) == 0 and coord[4] >= conf_threshold:
                x1, y1, x2, y2, conf = coord
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                
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
    input_path = 'crowd.mp4'
    output_path = 'woutput_crowd.mp4'
    
    model = load_model()
    total_people_count = process_video(input_path, output_path, model, conf_threshold=0.2)
    print(f'Общее количество распознанных людей: {total_people_count}') #9164 для Yolo5s 10003 для Yolo5x

if __name__ == "__main__":
    main()
