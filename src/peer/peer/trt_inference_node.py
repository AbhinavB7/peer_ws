import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
import os
import torch
from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_binding = self.engine[0]
        self.output_binding = self.engine[1]

        self.input_shape = self.engine.get_tensor_shape(self.input_binding)
        self.output_shape = self.engine.get_tensor_shape(self.output_binding)

        self.input_size = trt.volume(self.input_shape) * np.float32().nbytes
        self.output_size = trt.volume(self.output_shape) * np.float32().nbytes

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.bindings = [int(self.d_input), int(self.d_output)]

    def infer(self, input_np):
        input_np = np.ascontiguousarray(input_np.astype(np.float32))
        cuda.memcpy_htod(self.d_input, input_np)
        self.context.execute_v2(bindings=self.bindings)
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)
        return output


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.bridge = CvBridge()

        # Paths
        detect_path = os.path.join(
            get_package_share_directory('peer'), 'models/obj_detection/best.pt')
        segment_trt_path = os.path.join(
            get_package_share_directory('peer'), 'models/segmentation/mobilenet_fp16.trt')

        # Load YOLO 
        self.det_model = YOLO(detect_path)
        self.get_logger().info(f"Loaded YOLO model from: {detect_path}")

        # Load segmentation model
        self.seg_model = TRTModel(segment_trt_path)
        self.get_logger().info(f"Loaded segmentation model (TensorRT) from: {segment_trt_path}")

        # Preprocessing for segmentation
        self.seg_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.image_callback,
            qos_profile
        )

        self.pub_detect = self.create_publisher(Image, '/pallet_detection', 10)
        self.pub_segment = self.create_publisher(Image, '/ground_segmentation', 10)

        self.get_logger().info("Inference node ready!")

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # YOLOv8 
            detect_result = self.det_model.predict(source=img_rgb, verbose=False, conf=0.15)[0]
            img_with_boxes = detect_result.plot()

            # Segmentation (TensorRT) 
            augmented = self.seg_transform(image=img_rgb)
            seg_input = augmented['image'].unsqueeze(0).cpu().numpy()
            seg_output = self.seg_model.infer(seg_input)[0][0]

            mask = (seg_output > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            overlay_seg = img.copy()
            overlay_seg[mask == 255] = [0, 255, 0]

            self.pub_detect.publish(self.bridge.cv2_to_imgmsg(img_with_boxes, encoding='bgr8'))
            self.pub_segment.publish(self.bridge.cv2_to_imgmsg(overlay_seg, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
