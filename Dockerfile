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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements_docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt


# Optional: Install PyTorch with GPU (CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install ultralytics

# Optional: Install OpenCV with contrib (CPU version)
RUN pip install opencv-python-headless

WORKDIR /workspace

