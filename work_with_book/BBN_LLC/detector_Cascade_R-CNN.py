import cv2
import torch
import torchvision
from torchvision import transforms

def load_model():
    """
    Загружает предобученную модель Cascade R-CNN для детекции объектов.
    Возвращает модель.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def process_video(input_path, output_path, model, conf_threshold=0.25):
    """
    Обрабатывает видео, выполняя детекцию объектов и сохраняя результат.
    Считает количество распознанных объектов за все видео.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model: Модель Cascade R-CNN для детекции.
        conf_threshold (float): Порог уверенности для детекции.

    Returns:
        int: Общее количество распознанных объектов.
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_objects_count = 0
    transform = transforms.Compose([transforms.ToTensor()])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
        
        frame_objects_count = 0
        for i in range(len(outputs[0]['labels'])):
            if outputs[0]['scores'][i] >= conf_threshold:
                x1, y1, x2, y2 = outputs[0]['boxes'][i].cpu().numpy().astype(int)
                label = outputs[0]['labels'][i].item()
                score = outputs[0]['scores'][i].item()
                
                # Отрисовка прямоугольника и текста на кадре
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'Object {label} {score:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                frame_objects_count += 1
        
        total_objects_count += frame_objects_count
        out.write(frame)
    
    cap.release()
    out.release()
    
    return total_objects_count

def main():
    """
    Основная точка входа в программу.
    """
    input_path = 'work_with_book/BBN_LLC/video/crowd.mp4'
    output_path = 'work_with_book/BBN_LLC/video/output_crowd.mp4'
    
    model = load_model()
    total_objects_count = process_video(input_path, output_path, model, conf_threshold=0.2)
    print(f'Общее количество распознанных объектов: {total_objects_count}')

if __name__ == "__main__":
    main()
