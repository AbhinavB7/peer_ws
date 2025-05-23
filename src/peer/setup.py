import os 
from glob import glob 
from setuptools import setup 

package_name = 'peer'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/inference.launch.py']),
        *[(os.path.join('share', package_name, os.path.dirname(f)), [f])
          for f in glob('scripts/**/*', recursive=True) if os.path.isfile(f)],
        *[(os.path.join('share', package_name, os.path.dirname(f)), [f])
          for f in glob('models/**/*', recursive=True) if os.path.isfile(f)],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhinav Bhamidipati',
    maintainer_email='abhinav7@terpmail.umd.edu',
    description='ROS2 inference node for pallet detection and ground segmentation',
    entry_points={
        'console_scripts': [
            'inference_node = peer.ros_inference_node:main',
            'trt_node = peer.trt_inference_node:main',
        ],
    },
)