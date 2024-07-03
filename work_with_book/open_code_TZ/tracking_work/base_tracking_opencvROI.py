import cv2
import sys
import os

# Список доступных трекеров
tracker_types = [ 'KCF',]
# 'BOOSTING', 'MIL'
# 'KCF', 'TLD', 'MEDIANFLOW','MOSSE', 'CSRT'
# Функция для создания трекера по его имени
def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        raise ValueError("Unknown tracker type")

# Основная функция
def track_objects(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проверка существования видеофайла
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return
    
    for tracker_type in tracker_types:
        # Создание трекера
        tracker = create_tracker(tracker_type)

        # Чтение видео
        video = cv2.VideoCapture(video_path)

        # Выход, если видео не открылось
        if not video.isOpened():
            print(f"Could not open video {video_path}")
            sys.exit()

        # Чтение первого кадра
        ok, frame = video.read()
        if not ok:
            print(f"Cannot read video file {video_path}")
            sys.exit()

        # Определение начальной рамки отслеживания
        bbox = cv2.selectROI(frame, False)
        if not bbox:
            print(f"Could not select ROI for tracker {tracker_type}")
            continue

        # Инициализация трекера с первого кадра и рамки отслеживания
        ok = tracker.init(frame, bbox)

        # Определение кодека и создание объекта VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(output_dir, f"{tracker_type}_output.avi")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        # Список для хранения предыдущих центров
        centers = []

        while True:
            # Чтение нового кадра
            ok, frame = video.read()
            if not ok:
                break

            # Обновление трекера
            ok, bbox = tracker.update(frame)

            # Рисование рамки и линии
            if ok:
                # Трекинг успешен
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                
                # Рисование линии от центра
                center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                centers.append(center)
                for i in range(1, len(centers)):
                    cv2.line(frame, centers[i - 1], centers[i], (0, 255, 0), 2)
            else:
                # Трекинг провален
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Отображение типа трекера на кадре
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Запись кадра в выходной видеофайл
            out.write(frame)

            # Отображение результата
            cv2.imshow("Tracking", frame)

            # Выход по клавише 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()

# Путь к видео и директория для выходных файлов
video_path = r"C:\Users\Alex\Desktop\test_work_with_neyron\drafts_AI\work_with_book\open_code_TZ\real_TZ\video\video_1_runners.mp4"
output_dir = "tracking_results"

track_objects(video_path, output_dir)


