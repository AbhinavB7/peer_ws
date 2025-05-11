# Peer Robotics 

## Directory Structure
```
└── src
    └── peer
        ├── annotate_scripts/
        │   └── ground_segment/
        ├── datasets/
        │   ├── loco_dataset/
        │   │   ├── images/ (train/val/test)
        │   │   └── labels/ (train/val/test)
        │   └── Segmentation/ (train/val/test)
        ├── launch/
        ├── models/
        │   ├── obj_detection/best.pt
        │   └── segmentation/
        │       ├── mobilenet.pth
        │       └── mobilenet_fp16.trt
        ├── peer/
        │   ├── ros_inference_node.py
        │   └── trt_inference_node.py
        ├── resource/
        ├── rviz/
        └── scripts/
            ├── onnx conversion/
            ├── runs/
            │   ├── detect/pallet_with_loco/weights/
            │   └── segmentation/
            ├── train_obj_model.py
            ├── test_obj_model.py
            ├── train_segment_model.py
            └── test_segment_model.py

```
## Models Link

```
https://drive.google.com/drive/folders/1GbwXGTLFWvfE-Yg82ANCKYy_36ZPvyNN?usp=sharing
```

## Installation

**Option 1: Native Setup**
```bash
pip install -r requirements.txt
```

**Option 2: Docker Build**
```bash
docker build -t ros2-peer-gpu .
```

## Running the ROS 2 Inference Node

### Native Workflow
```bash
colcon build
source install/setup.bash
ros2 launch peer inference.launch.py
```

### Docker Workflow
```bash
docker run --gpus all -it --name ros2-peer-container \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  -v ~/peer_ws:/workspace \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --env="CUDA_VISIBLE_DEVICES=0" \
  ros2-peer-gpu:latest bash

```

After starting the container, run:
```bash
colcon build
source install/setup.bash
ros2 launch peer inference.launch.py
```

## Notes
- YOLOv8 model runs in PyTorch due to TensorRT conversion issues but performs fast enough for deployment.
- Ground segmentation model is optimized and runs using TensorRT in FP16.
- The node subscribes to RGB and depth topics and publishes overlaid outputs on `/pallet_detection` and `/ground_segmentation`.

## Training and Testing Models

### Object Detection (YOLOv8)
- To train the YOLOv8 model, use:
  ```bash
  python3 scripts/train_obj_model.py
  ```
  Make sure to update dataset paths inside the script.

- To test and evaluate the YOLOv8 model, use:
  ```bash
  python3 scripts/test_obj_model.py
  ```

### Ground Segmentation (UNet)
- To train the segmentation model:
  ```bash
  python3 scripts/train_segment_model.py
  ```

- To test the segmentation model:
  ```bash
  python3 scripts/test_segment_model.py
  ```

## ROS Inference Nodes

- The `peer/` folder contains two ROS 2 inference nodes:
  - `ros_inference_node.py`: Runs YOLOv8 and UNet using PyTorch `.pt` models.
  - `trt_inference_node.py`: Runs YOLOv8 in PyTorch and UNet as a TensorRT engine (`mobilenet_fp16.trt`).

## Annotation Tools

- The `annotate_scripts/ground_segment/` folder includes tools to generate masks:
  - `masks.py` and `segment.py` use GroundingDINO and SAM to create segmentation masks for training.

Be sure to verify all file paths before running training or inference scripts.

## TensorRT Conversion

- For ONNX conversion use `scripts/onnx_conversion/onnx_conversion.py`.
- Convert that onnx to trt using 
```
trtexec --onnx=mobilenet.onnx --saveEngine=model_fp16_new.trt --fp16
```