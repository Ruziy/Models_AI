import cv2
import numpy as np
import torch

def load_model(weights_path='yolov4x-mish.weights', config_path='yolov4x-mish.cfg'):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def process_video(input_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео, выполняя детекцию людей и сохраняя результат.
    Считает количество распознанных людей за все видео.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model: Модель YOLOv4 для детекции.
        конф_THRESHOLD (float): Порог уверенности для детекции.

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

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward(model.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
        frame_people_count = 0

        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            if class_ids[i] == 0:  # Assuming class_id 0 is 'person' for YOLOv4
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f'Person {confidences[i]:.2f}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    input_path = 'path_to_input_video.mp4'
    output_path = 'path_to_output_video.mp4'
    
    model = load_model(weights_path='yolov4x-mish.weights', config_path='yolov4x-mish.cfg')
    total_people_count = process_video(input_path, output_path, model, conf_threshold=0.2)
    print(f'Общее количество распознанных людей: {total_people_count}')

if __name__ == "__main__":
    main()
