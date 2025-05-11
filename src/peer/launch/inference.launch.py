from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='peer',
            executable='trt_node',
            name='inference_node',
            output='screen'
        )
    ])
