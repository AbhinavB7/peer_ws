from ultralytics import YOLO
import os

#ad trained model 
model = YOLO("/home/abhi/peer_ws/src/peer/scripts/runs/detect/pallet_with_loco/weights/best.pt")

test_dir = "../datasets/Pallet_detect/Pallet_yolo/test/images"

# Run inference
results = model.predict(
    source=test_dir,         
    save=True,               
    save_txt=True,           
    project="runs/detect",   
    name="pallet_test_eval_with_loco_pth", 
    conf=0.25,               
    iou=0.5,                 
    device=0                 
)

# Evaluate performance on test set if labels exist
metrics = model.val(
    data="../datasets/Pallet_detect/Pallet_yolo/data.yaml", 
    split="test",
    conf=0.25,
    iou=0.5,
    plots=True,
    save_json=True
)

# Print metrics
print("\nEvaluation Metrics on Test Set:")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p[0]:.4f}")
print(f"Recall: {metrics.box.r[0]:.4f}")

