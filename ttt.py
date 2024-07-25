import cv2
import cvzone
from ultralytics import YOLO
import numpy as np
import math
from sort import *
import os

# Load video
cap = cv2.VideoCapture("iii.mp4")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Class names for YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Create directory to save cropped images if it doesn't exist
output_dir = "cropped_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tracking (using sort.py file)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

frame_count = 0

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, stream=True)

    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            currentarray = np.array([x1, y1, x2, y2, conf])
            detection = np.vstack((detection, currentarray))

    resultTracker = tracker.update(detection)

    for result in resultTracker:
        x1, y1, x2, y2, ID = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        currentclass = classNames[int(box.cls[0])]
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 255, 0))  # l: length of bounding box, rt: rectangle thickness
        cvzone.putTextRect(img, f'{int(ID)}, {currentclass}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=6)

        # Crop the detected object
        cropped_img = img[y1:y2, x1:x2]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"{currentclass}_{int(ID)}_{frame_count}.png")
        cv2.imwrite(output_path, cropped_img)

    cv2.imshow("Image", img)
    frame_count += 1

    cv2.waitKey(1) 

cap.release()
cv2.destroyAllWindows()
