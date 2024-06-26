import cv2
import torch
import torchvision
from torchvision import transforms
import multiprocessing as mp
import time

# Глобальная модель для всех процессов
model = None

def load_model():
    """
    Загружает предобученную модель SSD300 VGG16 для детекции людей.
    Возвращает модель.
    """
    global model
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()

def init_model():
    """
    Инициализирует модель в каждом процессе.
    """
    global model
    if model is None:
        load_model()

def process_frame(frame, conf_threshold=0.2):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)

    for i in range(len(outputs[0]['labels'])):
        if outputs[0]['labels'][i] == 1 and outputs[0]['scores'][i] >= conf_threshold:
            x1, y1, x2, y2 = outputs[0]['boxes'][i].numpy().astype(int)
            conf = outputs[0]['scores'][i].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'Person {conf:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def process_video(input_path, output_path, conf_threshold=0.2):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pool = mp.Pool(mp.cpu_count(), initializer=init_model)
    frame_count = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f'Обрабатывается кадр номер: {frame_count}')

        results.append(pool.apply_async(process_frame, args=(frame, conf_threshold)))

    for result in results:
        frame = result.get()
        out.write(frame)

    cap.release()
    out.release()
    pool.close()
    pool.join()

def main():
    """
    Основная точка входа в программу.
    """
    input_path = 'crowd.mp4'
    output_path = 'output_crowd.mp4'

    start_time = time.time()
    load_model()
    process_video(input_path, output_path, conf_threshold=0.2)
    end_time = time.time()
    time_without_segmentation = end_time - start_time
    print(f"Time with segmentation: {time_without_segmentation:.2f} seconds")



if __name__ == "__main__":
    main()