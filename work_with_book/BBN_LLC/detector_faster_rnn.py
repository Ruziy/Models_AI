import cv2
import torch
import torchvision
from torchvision import transforms
import time
import multiprocessing as mp

def load_model(device='cpu'):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    return model

def process_frame(frame,model,conf_threshold=0.2):
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

def process_video(input_path, output_path, model, conf_threshold=0.2):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pool = mp.Pool(mp.cpu_count())

        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results.append(pool.apply_async(process_frame, args=(frame,model,conf_threshold)))

        for result in results:
            frame = result.get()
            out.write(frame)


        cap.release()
        out.release()
        pool.close()
        pool.join()


def main():
    input_path = 'output_crowd_reencoded.mp4'  # Используем перекодированное видео
    output_path = 'output_crowd.mp4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(device)
    start_time = time.time()
    process_video(input_path, output_path, model, conf_threshold=0.2)
    end_time = time.time()
    time_with_segmentation = end_time - start_time
    print(f"Time with segmentation: {time_with_segmentation:.2f} seconds")

    with open('processing_time.txt', 'w') as file:
        file.write(f"Time with segmentation: {time_with_segmentation:.2f} seconds\n")


if __name__ == "__main__":
    main()
