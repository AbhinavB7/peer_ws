import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
import os


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.bridge = CvBridge()
        self.depth_image = None

        model_path = os.path.join(
            get_package_share_directory('peer'),
            'models/obj_detection/best.pt'
        )

        segment_path = os.path.join(
            get_package_share_directory('peer'),
            'models/segmentation/mobilenet.pth'
        )

        self.declare_parameter("model_detect_path", model_path)
        self.declare_parameter("model_segment_path", segment_path)

        # Load YOLOv8 model
        detect_path = self.get_parameter("model_detect_path").get_parameter_value().string_value
        self.det_model = YOLO(detect_path)
        self.get_logger().info(f"Loaded detection model from: {detect_path}")

        # Load segmentation model
        segment_path = self.get_parameter("model_segment_path").get_parameter_value().string_value
        # self.seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
        self.seg_model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=1, activation=None)

        self.seg_model.load_state_dict(torch.load(segment_path, map_location='cuda'))
        self.seg_model.eval().cuda()
        self.get_logger().info(f"Loaded segmentation model from: {segment_path}")

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
        
        self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_depth',
            self.depth_callback,
            qos_profile
        )
        
        # Publishers
        self.pub_detect = self.create_publisher(Image, '/pallet_detection', 10)
        self.pub_segment = self.create_publisher(Image, '/ground_segmentation', 10)

        self.get_logger().info("Inference node ready!")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth callback error: {e}")


    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 inference
            detect_result = self.det_model.predict(source=img_rgb, verbose=False, conf=0.15)[0]
            img_with_boxes = detect_result.plot()

            # Run segmentation
            augmented = self.seg_transform(image=img_rgb)
            tensor = augmented['image'].unsqueeze(0).cuda()
            with torch.no_grad():
                pred_mask = torch.sigmoid(self.seg_model(tensor)).squeeze().cpu().numpy()
                mask = (pred_mask > 0.5).astype(np.uint8) * 255
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            overlay_seg = img.copy()
            overlay_seg[mask == 255] = [0, 255, 0]  # Green for ground

            # Publish results
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
