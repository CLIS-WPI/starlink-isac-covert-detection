# Base image: NVIDIA TensorFlow
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /workspace

# Set timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

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

# Copy project files
COPY . .

# CMD: print versions + CUDA/cuDNN info + enter bash
CMD ["bash", "-c", "\
echo '=== Installed Versions ==='; \
python3 -c 'import sys, tensorflow as tf, sionna; import importlib.metadata as m; \
print(f\"Python: {sys.version.split()[0]}\"); \
print(f\"TensorFlow: {tf.__version__}\"); \
print(f\"Sionna: {sionna.__version__}\"); \
print(f\"OpenNTN: {m.version(\"openntn\")}\")'; \
echo '=== CUDA/cuDNN ==='; \
nvcc --version; \
python3 -c 'import tensorflow as tf; \
print(f\"TF built with CUDA: {tf.test.is_built_with_cuda()}\"); \
print(f\"TF GPU Available: {tf.config.list_physical_devices(\"GPU\")}\")'; \
exec bash"]

