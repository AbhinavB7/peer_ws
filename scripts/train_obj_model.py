from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

# Train
model.train(
    data="../datasets/Pallet_detect/Pallet_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="pallet_detector_yolov8n",
    device=0  
)
