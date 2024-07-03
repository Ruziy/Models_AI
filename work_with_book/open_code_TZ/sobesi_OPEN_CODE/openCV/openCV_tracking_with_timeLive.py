import cv2
import numpy as np

def load_classes(filepath):
    """
    Загрузка классов объектов из файла.

    Args:
    filepath (str): Путь к файлу с именами классов.

    Returns:
    list: Список имен классов.
    """
    with open(filepath, 'r') as f:
        return f.read().strip().split('\n')

def load_yolo_model(weights, cfg):
    """
    Загрузка модели YOLO и получение выходных слоев.

    Args:
    weights (str): Путь к файлу весов модели YOLO.
    cfg (str): Путь к файлу конфигурации модели YOLO.

    Returns:
    net: Загрузка модели YOLO.
    list: Список имен выходных слоев.
    """
    net = cv2.dnn.readNet(weights, cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def get_blob(frame):
    """
    Преобразование изображения в формат blob.

    Args:
    frame (numpy.ndarray): Изображение.

    Returns:
    blob: Преобразованное изображение.
    """
    return cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)

def get_detections(net, output_layers, blob):
    """
    Получение детекций из сети YOLO.

    Args:
    net: Загрузка модели YOLO.
    output_layers (list): Список выходных слоев.
    blob: Преобразованное изображение.

    Returns:
    list: Результаты детекций.
    """
    net.setInput(blob)
    return net.forward(output_layers)

def process_detections(outs, width, height):
    """
    Обработка детекций для получения bounding boxes, уверенности и идентификаторов классов.

    Args:
    outs (list): Результаты детекций.
    width (int): Ширина изображения.
    height (int): Высота изображения.

    Returns:
    list: Список bounding boxes.
    list: Список уверенности.
    list: Список идентификаторов классов.
    """
    boxes = []
    confidences = []
    class_ids = []
    for out_1 in outs:
        for detection in out_1:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def apply_nms(boxes, confidences):
    """
    Применение non-max suppression для исключения перекрывающихся боксов.

    Args:
    boxes (list): Список bounding boxes.
    confidences (list): Список уверенности.

    Returns:
    list: Список индексов выбранных боксов после применения non-max suppression.
    """
    return cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

def initialize_trackers(boxes, indexes, frame):
    """
    Инициализация трекеров для каждого обнаруженного объекта.

    Args:
    boxes (list): Список bounding boxes.
    indexes (list): Список индексов выбранных боксов.
    frame (numpy.ndarray): Изображение.

    Returns:
    list: Список инициализированных трекеров.
    """
    trackers = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)
    return trackers

def update_trackers(trackers, frame):
    """
    Обновление трекеров и возврат успешно обновленных.

    Args:
    trackers (list): Список трекеров.
    frame (numpy.ndarray): Изображение.

    Returns:
    list: Список успешно обновленных трекеров и их bounding boxes.
    """
    new_trackers = []
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            new_trackers.append((tracker, box))
    return new_trackers

def check_and_restore_objects(boxes, indexes, dead_objects, frame, max_life_time):
    """
    Проверка и восстановление объектов из мертвых объектов.

    Args:
    boxes (list): Список bounding boxes.
    indexes (list): Список индексов выбранных боксов.
    dead_objects (dict): Словарь мертвых объектов.
    frame (numpy.ndarray): Изображение.
    max_life_time (int): Максимальное время жизни объектов.

    Returns:
    list: Список восстановленных трекеров.
    list: Список восстановленных идентификаторов классов.
    list: Список восстановленных траекторий.
    """
    restored_trackers = []
    restored_class_ids = []
    restored_trajectories = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center = (x + w // 2, y + h // 2)
            for obj_id in list(dead_objects.keys()):
                tracker = dead_objects[obj_id]['tracker']
                success, box = tracker.update(frame)
                if success:
                    prev_center = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
                    distance = np.linalg.norm(np.array(center) - np.array(prev_center))
                    if distance < 50:
                        restored_trackers.append(tracker)
                        restored_class_ids.append(dead_objects[obj_id]['class_id'])
                        traj = dead_objects[obj_id]['trajectory']
                        traj.append(center)
                        restored_trajectories.append(traj)
                        del dead_objects[obj_id]
                        break
    return restored_trackers, restored_class_ids, restored_trajectories

def update_dead_objects(dead_objects):
    """
    Обновление времени жизни мертвых объектов и удаление объектов с истекшим временем жизни.

    Args:
    dead_objects (dict): Словарь мертвых объектов.
    """
    dead_objects_to_remove = []
    for obj_id in dead_objects:
        dead_objects[obj_id]['life_time'] -= 1
        if dead_objects[obj_id]['life_time'] <= 0:
            dead_objects_to_remove.append(obj_id)
    for obj_id in dead_objects_to_remove:
        del dead_objects[obj_id]

def draw_objects(frame, trackers, class_ids, trajectories, classes):
    """
    Отрисовка bounding boxes и траекторий объектов на кадре.

    Args:
    frame (numpy.ndarray): Изображение.
    trackers (list): Список трекеров и их bounding boxes.
    class_ids (list): Список идентификаторов классов.
    trajectories (list): Список траекторий объектов.
    classes (list): Список имен классов.
    """
    for i, (tracker, box) in enumerate(trackers):
        x, y, w, h = [int(v) for v in box]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        center = (x + w // 2, y + h // 2)
        
        trajectories[i].append(center)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for j in range(1, len(trajectories[i])):
            if trajectories[i][j - 1] is None or trajectories[i][j] is None:
                continue
            cv2.line(frame, trajectories[i][j - 1], trajectories[i][j], color, 2)

def main():
    """
    Основная функция, которая объединяет все части вместе и выполняет трекинг объектов на видео.
    """
    classes = load_classes('models_params/coco.names')
    net, output_layers = load_yolo_model('models_params/yolov3.weights', 'models_params/yolov3.cfg')
    
    cap = cv2.VideoCapture('video/video_1.mp4')
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_file = 'video/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Ошибка при инициализации VideoWriter")
        return

    trackers = []
    class_ids = []
    trajectories = []
    dead_objects = {}
    max_life_time = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = get_blob(frame)
        outs = get_detections(net, output_layers, blob)
        boxes, confidences, temp_class_ids = process_detections(outs, width, height)
        indexes = apply_nms(boxes, confidences)

        previous_trackers = trackers.copy()
        previous_class_ids = class_ids.copy()
        previous_trajectories = trajectories.copy()

        trackers = initialize_trackers(boxes, indexes, frame)
        class_ids = [temp_class_ids[i] for i in range(len(boxes)) if i in indexes]
        trajectories = [[(boxes[i][0] + boxes[i][2] // 2, boxes[i][1] + boxes[i][3] // 2)] for i in range(len(boxes)) if i in indexes]

        for j, (tracker, cls_id, traj) in enumerate(zip(previous_trackers, previous_class_ids, previous_trajectories)):
            if tracker not in trackers:
                dead_objects[len(dead_objects)] = {'tracker': tracker, 'class_id': cls_id, 'trajectory': traj, 'life_time': max_life_time}

        update_dead_objects(dead_objects)
        restored_trackers, restored_class_ids, restored_trajectories = check_and_restore_objects(boxes, indexes, dead_objects, frame, max_life_time)

        trackers.extend(restored_trackers)
        class_ids.extend(restored_class_ids)
        trajectories.extend(restored_trajectories)

        tracked_objects = update_trackers(trackers, frame)
        draw_objects(frame, tracked_objects, class_ids, trajectories, classes)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

