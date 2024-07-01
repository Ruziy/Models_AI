import cv2
import numpy as np

# Загрузка видеопотока
cap = cv2.VideoCapture('video.mp4')

# Инициализация детектора объектов (например, использование фона для простоты)
fgbg = cv2.createBackgroundSubtractorMOG2()

# Инициализация трекера объектов
tracker = cv2.MultiTracker_create()

# Инициализация списка для хранения траекторий
trajectories = {}

# Главный цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Обнаружение объектов
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Обновление трекеров
    if len(tracker.getObjects()) == 0:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            tracker.add(cv2.TrackerCSRT_create(), frame, (x, y, w, h))
    
    success, boxes = tracker.update(frame)
    
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (200, 0, 0), 2, 1)
        
        # Обновление траекторий
        center = (int(newbox[0] + newbox[2] / 2), int(newbox[1] + newbox[3] / 2))
        if i not in trajectories:
            trajectories[i] = []
        trajectories[i].append(center)
        
        # Анализ траектории
        if len(trajectories[i]) > 10:
            speed = np.linalg.norm(np.array(trajectories[i][-1]) - np.array(trajectories[i][-2]))
            if speed > threshold_speed:  # Порог скорости
                cv2.putText(frame, "Anomalous", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Отображение кадра
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
