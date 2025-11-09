# Base image: NVIDIA TensorFlow
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /workspace

# Set timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget vim bash tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-minimal.txt .

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements-minimal.txt && \
    python3 -m pip install --no-cache-dir sionna==1.2.1

# Clone and install OpenNTN (main branch)
RUN git clone -b main https://github.com/ant-uni-bremen/OpenNTN.git /opt/OpenNTN && \
    cd /opt/OpenNTN && \
    chmod +x install.sh && \
    . ./install.sh

# Copy project files (mounted later by devcontainer)
COPY . .

# 🔹 [Optional but recommended] keep TensorFlow logs clean
ENV TF_CPP_MIN_LOG_LEVEL=2

# ✅ Final CMD: Start an interactive shell instead of script
# (So Cursor can attach and execute commands inside container)
CMD ["/bin/bash"]
