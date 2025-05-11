# from ultralytics import YOLO

# # ==== Step 1: Load Trained YOLOv8 Model ====
# model_path = "/home/abhi/peer_ws/src/peer/scripts/runs/detect/pallet_with_loco/weights/best.pt"  # Path to your trained model
# model = YOLO(model_path)

# # ==== Step 2: Export to ONNX ====
# model.export(
#     format="onnx", 
#     opset=12, 
#     simplify=True, 
#     imgsz=(640, 640)  # Must match training image size
# )

# print("YOLO model exported to ONNX format.")

# import torch
# import segmentation_models_pytorch as smp

# # === Step 1: Define and load model ===
# model = smp.Unet(
#     encoder_name="mobilenet_v2", 
#     in_channels=3, 
#     classes=1, 
#     activation=None
# )
# model.load_state_dict(torch.load("/home/abhi/peer_ws/src/peer/models/segmentation/mobilenet.pth", map_location="cpu"))
# model.eval()

# # === Step 2: Create dummy input ===
# dummy_input = torch.randn(1, 3, 256, 256)  # Match training size

# # === Step 3: Export to ONNX ===
# torch.onnx.export(
#     model,
#     dummy_input,
#     "mobilenet.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=12,
#     do_constant_folding=True,
# )

# print("MobileNet segmentation model exported to ONNX (256x256 input).")

import subprocess
from ultralytics import YOLO

# === CONFIG ===
pt_path = "/home/abhi/peer_ws/src/peer/scripts/runs/detect/pallet_with_loco/weights/best.pt"
onnx_path = "best.onnx"
engine_path = "best_fp16.trt"
imgsz = 640

# === STEP 1: Export to ONNX ===
print("[INFO] Exporting YOLO to ONNX...")
model = YOLO(pt_path)
model.export(format="onnx", opset=12, simplify=True, imgsz=imgsz, dynamic=False)
print(f"[✅] Exported to: {onnx_path}")

# === STEP 2: Convert to TensorRT ===
print("[INFO] Converting ONNX to TensorRT engine (FP16)...")
trt_cmd = [
    "trtexec",
    f"--onnx={onnx_path}",
    f"--saveEngine={engine_path}",
    "--fp16",
    f"--explicitBatch"
]

try:
    subprocess.run(trt_cmd, check=True)
    print(f"[✅] Saved TensorRT engine: {engine_path}")
except subprocess.CalledProcessError as e:
    print("[❌] TensorRT conversion failed")
    print(e)

