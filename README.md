# Covert Channel Detection in LEO Satellite ISAC Systems

**Real-Time Covert Leakage Detection in Large-Scale LEO Satellite Networks**

ML-based detector for covert signals in LEO satellites using 3GPP TR38.811 NTN channel models with deep learning (CNN).

## 🎯 Overview

This project implements an intelligent and automated covert channel detection system for communication and satellite systems (ISAC/NTN) using deep learning. The system can detect when a transmitter injects hidden messages into the main signal without significant changes in power or waveform appearance.

### Key Features

- **Deep Learning Detection**: CNN-based detector that learns spectral patterns in OFDM signals
- **Realistic Simulation**: Uses Sionna and OpenNTN for realistic satellite channel modeling
- **Covert Injection**: Power-preserving covert channel injection that maintains signal appearance
- **Multi-GPU Support**: Parallel dataset generation using 2 GPUs
- **CI/CD Integration**: Automated testing and Docker builds

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA Container Toolkit
- 2x GPUs (for parallel dataset generation)

### Run with Docker

```bash
# Build image
docker build -t covert_l -f .devcontainer/Dockerfile .

# Run container (with GPU support)
docker run --gpus all -it --user root \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --name covert_l_dev \
  -v "$(pwd)":/workspace \
  -w /workspace \
  covert_l:latest

docker start -ai covert_l_dev


# Generate dataset (parallel on 2 GPUs)
python3 generate_dataset_parallel.py

# Train CNN detector
python3 main_detection_cnn.py --epochs 50 --batch-size 512

# With CSI fusion
python3 main_detection_cnn.py --use-csi --epochs 50
```

## 📁 Project Structure

```
.
├── config/          # Configuration files
│   └── settings.py  # Main configuration (COVERT_AMP, FOCAL_LOSS, etc.)
├── core/            # Core modules
│   ├── isac_system.py        # ISAC system with NTN channel models
│   ├── dataset_generator.py  # Multi-satellite dataset generation
│   └── covert_injection.py   # Covert channel injection logic
├── model/           # Detection models
│   ├── detector_cnn.py        # CNN-based detector (main)
│   └── detector_frequency.py # RandomForest detector (alternative)
├── utils/           # Utility functions
├── tests/           # Unit tests
├── dataset/         # Generated datasets (created at runtime)
├── model/           # Saved models (created at runtime)
└── result/          # Results and metrics (created at runtime)
```

## ⚙️ Configuration

Key parameters in `config/settings.py`:

- `COVERT_AMP`: Covert signal amplitude (default: 0.7)
- `NUM_SAMPLES_PER_CLASS`: Dataset size per class (default: 500)
- `USE_FOCAL_LOSS`: Enable focal loss for hard examples (default: True)
- `FOCAL_LOSS_GAMMA`: Focus parameter (default: 2.5)
- `FOCAL_LOSS_ALPHA`: Class weighting (default: 0.5)

## 🔧 Usage

### Dataset Generation

```bash
python3 generate_dataset_parallel.py
```

This will:
- Generate dataset using 2 GPUs in parallel
- Cache NTN topologies for faster subsequent runs
- Save dataset to `dataset/` directory

### Training CNN Detector

```bash
# Basic training
python3 main_detection_cnn.py --epochs 50 --batch-size 512

# With CSI fusion
python3 main_detection_cnn.py --use-csi --epochs 50

# Multi-GPU training (Note: Sionna only supports single GPU, but dataset generation uses 2 GPUs)
python3 main_detection_cnn.py --multi-gpu
```

## 📊 Results

The pipeline outputs:
- Model saved to `model/cnn_detector.keras`
- Metrics (AUC, Precision, Recall, F1) saved to `result/detection_results_cnn.json`

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v
```

## 📝 Notes

- **GPU Limitation**: Sionna only supports single GPU for training, but dataset generation uses 2 GPUs via multiprocessing
- **Power Preservation**: Covert injection is power-preserving by default (configurable via `ABLATION_CONFIG`)
- **Pattern Injection**: Uses fixed pattern strategy for consistent CNN learning

## 🔗 Dependencies

See `requirements-minimal.txt` for full list. Key dependencies:
- Sionna 1.2.1
- TensorFlow 2.x
- OpenNTN (cloned and installed in Docker)
- scikit-learn
- numpy

## 📄 License

[Add your license here]

## 👥 Authors

[Add authors here]

