from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='peer',
            executable='inference_node',
            name='inference_node',
            parameters=[
                {"model_detect_path": "/home/abhi/peer_ws/src/peer/scripts/runs/detect/pallet_detector_yolov8n2-n/weights/best.pt"},
                {"model_segment_path": "/home/abhi/peer_ws/src/peer/models/segmentation/mobilenet.pth"}
            ],
            output='screen'
        )
    ])
