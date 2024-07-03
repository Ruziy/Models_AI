import cv2

try:
    print("OpenCV version:", cv2.__version__)
    # Используем метод cv2.TrackerKCF_create для OpenCV 4.2.0
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    for tracker_type in tracker_types:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
            print(f"{tracker_type} is succes")
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
            print(f"{tracker_type} is succes")
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            print(f"{tracker_type} is succes")

    print("KCF Tracker created successfully.")
except AttributeError as e:
    print("AttributeError:", e)
except ImportError as e:
    print("ImportError:", e)
