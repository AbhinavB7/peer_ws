FROM ros:humble-ros-base-jammy

# Install base ROS2 desktop (includes RViz, Gazebo, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop=0.10.0-1* \
    build-essential \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    libgl1-mesa-glx \
    curl \
    wget \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

# Install PyTorch with GPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install ultralytics

# Install OpenCV with contrib
RUN pip install opencv-python-headless

# Install TensorRT Python bindings
RUN pip install tensorrt --extra-index-url https://pypi.nvidia.com

# Now install pycuda
RUN pip install pycuda

WORKDIR /workspace

