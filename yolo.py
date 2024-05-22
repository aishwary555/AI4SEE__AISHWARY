from ultralytics import YOLO
import cv2
 
model = YOLO('yolov8l.pt')
results = model("cars.png", show=True)

cv2.waitKey(0)

