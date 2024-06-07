import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# STEP 2: Create a GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='open_code_TZ/detection_of_moves/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

def process_frame(image, recognition_result):
    """Process the frame, drawing hand landmarks and gestures."""
    if recognition_result.gestures and recognition_result.hand_landmarks:
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks

        # Draw the top gesture name and score
        cv2.putText(image, f"{top_gesture.category_name} ({top_gesture.score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for landmarks in hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
            ])
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return image

def create_results_cap():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            print("Could not read frame!")
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(image)

        # Process the frame
        img = process_frame(img, recognition_result)
        
        cv2.imshow("Gesture Recognition", img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_results_cap()
