import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import albumentations as A
from albumentations.pytorch import ToTensorV2


def decode_yolo_output(raw_output, conf_thresh=0.15):
    boxes_conf_cls = []

    for det in raw_output:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, obj_conf = det[:5]
        class_scores = det[5:]
        if len(class_scores) == 0:
            continue
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        conf = obj_conf * class_conf
        if conf < conf_thresh:
            continue
        boxes_conf_cls.append([x1, y1, x2, y2, conf, class_id])

    return np.array(boxes_conf_cls)


def postprocess_yolo_trt(yolo_output, img_shape, conf_thresh=0.15, iou_thresh=0.5):
    boxes = []
    confidences = []
    class_ids = []

    for det in yolo_output:
        if len(det) != 6:
            print(f"[WARN] Skipping invalid detection: {det}")
            continue
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_thresh:
            continue
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]

    final_dets = []
    for i in np.array(indices).flatten():
        box = boxes[i]
        final_dets.append((box, class_ids[i], confidences[i]))

    return final_dets


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

        detect_trt_path = os.path.join(
            get_package_share_directory('peer'), 'models/best_fp16.trt')
        segment_trt_path = os.path.join(
            get_package_share_directory('peer'), 'models/segmentation/mobilenet_fp16.trt')

        self.det_model = TRTModel(detect_trt_path)
        self.seg_model = TRTModel(segment_trt_path)

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

        self.get_logger().info("Inference node using TensorRT ready!")

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- YOLO Inference ---
            yolo_input = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
            yolo_input = yolo_input.transpose(2, 0, 1)[np.newaxis, :]
            yolo_output = self.det_model.infer(yolo_input)

            if len(yolo_output.shape) == 1:
                self.get_logger().warn(f"YOLO output shape: {yolo_output.shape}. Attempting reshape...")
                try:
                    yolo_output = yolo_output.reshape(-1, yolo_output.shape[0] // 8400)
                except Exception as e:
                    self.get_logger().error(f"Failed to reshape YOLO output: {e}")
                    return
            elif len(yolo_output.shape) == 3:
                yolo_output = yolo_output[0]

            # Decode raw output to get [x1, y1, x2, y2, conf, cls]
            decoded = decode_yolo_output(yolo_output, conf_thresh=0.15)

            # Rescale boxes
            if decoded.shape[1] == 6:
                decoded[:, [0, 2]] *= img.shape[1] / 640
                decoded[:, [1, 3]] *= img.shape[0] / 640
                detections = postprocess_yolo_trt(decoded, img.shape)
            else:
                self.get_logger().warn(f"Decoded output shape invalid: {decoded.shape}")
                detections = []

            # Draw boxes
            img_with_boxes = img.copy()
            for box, cls_id, conf in detections:
                x, y, w, h = box
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(img_with_boxes, f'{cls_id}: {conf:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- Segmentation Inference ---
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
