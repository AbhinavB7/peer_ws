from ultralytics import YOLO

model = YOLO("/home/abhi/peer_ws/src/peer/scripts/runs/detect/pallet_detector_yolov8n2-n/weights/best.pt")  

# Train
model.train(
    data="../datasets/loco_in_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="pallet_with_loco",
    device=0  
)
