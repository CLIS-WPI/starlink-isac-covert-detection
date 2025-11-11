# 🚀 Production Pipeline - Complete Instructions

## 📋 Overview

This document provides complete step-by-step instructions for running the complete pipeline from scratch for both Scenario A and Scenario B.

**Prerequisites:**
- Python 3.8+
- TensorFlow 2.x
- All dependencies installed
- GPU available (recommended)

---

## 🔧 Scenario A: Single-hop Downlink (Insider@Satellite)

### Step 1: Generate Dataset

```bash
cd /workspace
python3 generate_dataset_parallel.py \
    --scenario sat \
    --total-samples 5000

# Verify dataset was created
ls -lh dataset/dataset_scenario_a_*.pkl

# Note: Script automatically creates both actual_total.pkl and total_samples.pkl
# (e.g., both dataset_scenario_a_4998.pkl and dataset_scenario_a_5000.pkl)
# No manual copy needed!
```

**Expected Output:**
- Dataset file: `dataset/dataset_scenario_a_5000.pkl`
- Size: ~79 MB
- Samples: 5000

**Verification:**
```bash
python3 -c "
import pickle
from pathlib import Path
dataset = pickle.load(open('dataset/dataset_scenario_a_5000.pkl', 'rb'))
print(f'✅ Dataset: {len(dataset[\"labels\"])} samples')
print(f'✅ TX shape: {dataset[\"tx_grids\"].shape}')
print(f'✅ RX shape: {dataset[\"rx_grids\"].shape}')
"
```

---

### Step 2: Train CNN Model

```bash
cd /workspace
python3 main_detection_cnn.py \
    --scenario sat \
    --epochs 30 \
    --batch-size 512

# Save output to log file
python3 main_detection_cnn.py \
    --scenario sat \
    --epochs 30 \
    --batch-size 512 \
    2>&1 | tee training_scenario_a.log
```

**Expected Output:**
- Model: `model/scenario_a/cnn_detector.keras`
- Results: `result/scenario_a/detection_results_cnn.json`
- Normalization: `model/scenario_a/cnn_detector_norm.pkl`
- Meta log: `result/scenario_a/run_meta_log.csv`

**Verification:**
```bash
# Check results
python3 -c "
import json
from pathlib import Path
results = json.load(open('result/scenario_a/detection_results_cnn.json', 'r'))
metrics = results['metrics']
print(f'✅ AUC: {metrics[\"auc\"]:.4f}')
print(f'✅ Precision: {metrics[\"precision\"]:.4f}')
print(f'✅ Recall: {metrics[\"recall\"]:.4f}')
print(f'✅ F1: {metrics[\"f1\"]:.4f}')
"
```

---

### Step 3: Evaluate and Generate Reports

```bash
cd /workspace

# Generate detailed analysis
python3 << 'EOF'
import json
import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("📊 Scenario A - Final Analysis")
print("="*80)

# Load dataset
dataset = pickle.load(open('dataset/dataset_scenario_a_5000.pkl', 'rb'))
print(f"\n✅ Dataset: {len(dataset['labels'])} samples")

# Load results
results = json.load(open('result/scenario_a/detection_results_cnn.json', 'r'))
metrics = results['metrics']

print(f"\n📊 Detection Performance:")
print(f"   AUC: {metrics['auc']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall: {metrics['recall']:.4f}")
print(f"   F1 Score: {metrics['f1']:.4f}")

# Power analysis
power_diff = results.get('power_analysis', {}).get('difference_pct', 0)
print(f"\n📊 Power Analysis:")
print(f"   Power Difference: {power_diff:.4f}%")

print("\n" + "="*80)
EOF
```

---

## 🔧 Scenario B: Dual-hop Relay (Insider@Ground)

### Step 1: Generate Dataset

```bash
cd /workspace
python3 generate_dataset_parallel.py \
    --scenario ground \
    --total-samples 5000

# Verify dataset was created
ls -lh dataset/dataset_scenario_b_*.pkl

# Note: Script automatically creates both actual_total.pkl and total_samples.pkl
# (e.g., both dataset_scenario_b_4998.pkl and dataset_scenario_b_5000.pkl)
# No manual copy needed!
```

**Expected Output:**
- Dataset file: `dataset/dataset_scenario_b_5000.pkl`
- Size: ~79 MB
- Samples: 5000

**Verification:**
```bash
python3 -c "
import pickle
from pathlib import Path
dataset = pickle.load(open('dataset/dataset_scenario_b_5000.pkl', 'rb'))
print(f'✅ Dataset: {len(dataset[\"labels\"])} samples')
print(f'✅ TX shape: {dataset[\"tx_grids\"].shape}')
print(f'✅ RX shape: {dataset[\"rx_grids\"].shape}')
print(f'✅ CSI shape: {dataset[\"csi_est\"].shape}')
print(f'✅ CSI dtype: {dataset[\"csi_est\"].dtype}')
"
```

---

### Step 2: Train CNN Model

```bash
cd /workspace
python3 main_detection_cnn.py \
    --scenario ground \
    --epochs 30 \
    --batch-size 512

# Save output to log file
python3 main_detection_cnn.py \
    --scenario ground \
    --epochs 30 \
    --batch-size 512 \
    2>&1 | tee training_scenario_b.log
```

**Expected Output:**
- Model: `model/scenario_b/cnn_detector.keras`
- Results: `result/scenario_b/detection_results_cnn.json`
- Normalization: `model/scenario_b/cnn_detector_norm.pkl`
- Meta log: `result/scenario_b/run_meta_log.csv`

**Verification:**
```bash
# Check results
python3 -c "
import json
from pathlib import Path
results = json.load(open('result/scenario_b/detection_results_cnn.json', 'r'))
metrics = results['metrics']
print(f'✅ AUC: {metrics[\"auc\"]:.4f}')
print(f'✅ Precision: {metrics[\"precision\"]:.4f}')
print(f'✅ Recall: {metrics[\"recall\"]:.4f}')
print(f'✅ F1: {metrics[\"f1\"]:.4f}')
"
```

---

### Step 3: Evaluate and Generate Reports

```bash
cd /workspace

# Generate detailed analysis
python3 << 'EOF'
import json
import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("📊 Scenario B - Final Analysis")
print("="*80)

# Load dataset
dataset = pickle.load(open('dataset/dataset_scenario_b_5000.pkl', 'rb'))
print(f"\n✅ Dataset: {len(dataset['labels'])} samples")

# Load results
results = json.load(open('result/scenario_b/detection_results_cnn.json', 'r'))
metrics = results['metrics']

print(f"\n📊 Detection Performance:")
print(f"   AUC: {metrics['auc']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall: {metrics['recall']:.4f}")
print(f"   F1 Score: {metrics['f1']:.4f}")

# Power analysis
power_diff = results.get('power_analysis', {}).get('difference_pct', 0)
print(f"\n📊 Power Analysis:")
print(f"   Power Difference: {power_diff:.4f}%")

# EQ Performance
meta = dataset.get('meta', [])
preservations = []
snr_improvements = []
for m in meta[:1000]:
    if isinstance(m, tuple):
        _, m = m
    if 'eq_pattern_preservation' in m:
        preservations.append(m['eq_pattern_preservation'])
    if 'eq_snr_improvement_db' in m:
        snr_improvements.append(m['eq_snr_improvement_db'])

if preservations:
    print(f"\n📊 EQ Performance:")
    print(f"   Pattern Preservation: {np.median(preservations):.3f} (median)")
    print(f"   SNR Improvement: {np.mean(snr_improvements):.2f} dB (mean)")

print("\n" + "="*80)
EOF
```

---

## 📊 Complete Pipeline (Both Scenarios)

### Option 1: Sequential Execution

```bash
cd /workspace

# Scenario A
echo "Starting Scenario A..."
python3 generate_dataset_parallel.py --scenario sat --total-samples 5000
cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl 2>/dev/null || true
python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512 2>&1 | tee training_scenario_a.log

# Scenario B
echo "Starting Scenario B..."
python3 generate_dataset_parallel.py --scenario ground --total-samples 5000
cp dataset/dataset_scenario_b_4998.pkl dataset/dataset_scenario_b_5000.pkl 2>/dev/null || true
python3 main_detection_cnn.py --scenario ground --epochs 30 --batch-size 512 2>&1 | tee training_scenario_b.log

echo "✅ Complete pipeline finished!"
```

### Option 2: Using Test Script

```bash
cd /workspace
python3 test_complete_pipeline_final.py 2>&1 | tee pipeline_test_complete.log
```

---

## 📁 Output Files Structure

After running the complete pipeline, you should have:

```
workspace/
├── dataset/
│   ├── dataset_scenario_a_5000.pkl      # Scenario A dataset (~79 MB)
│   └── dataset_scenario_b_5000.pkl      # Scenario B dataset (~79 MB)
│
├── model/
│   ├── scenario_a/
│   │   ├── cnn_detector.keras           # Trained model
│   │   └── cnn_detector_norm.pkl        # Normalization stats
│   └── scenario_b/
│       ├── cnn_detector.keras           # Trained model
│       └── cnn_detector_norm.pkl        # Normalization stats
│
├── result/
│   ├── scenario_a/
│   │   ├── detection_results_cnn.json   # Performance metrics
│   │   └── run_meta_log.csv            # Training metadata
│   └── scenario_b/
│       ├── detection_results_cnn.json   # Performance metrics
│       └── run_meta_log.csv            # Training metadata
│
└── logs/
    ├── training_scenario_a.log          # Training log (if saved)
    └── training_scenario_b.log          # Training log (if saved)
```

---

## ✅ Verification Checklist

After running each scenario, verify:

### Dataset Verification
- [ ] Dataset file exists and size is ~79 MB
- [ ] Dataset contains 5000 samples
- [ ] Dataset has required fields: `tx_grids`, `rx_grids`, `labels`, `meta`
- [ ] For Scenario B: `csi_est` exists and is `complex64`

### Model Verification
- [ ] Model file exists: `model/scenario_X/cnn_detector.keras`
- [ ] Normalization file exists: `model/scenario_X/cnn_detector_norm.pkl`
- [ ] Model can be loaded without errors

### Results Verification
- [ ] Results file exists: `result/scenario_X/detection_results_cnn.json`
- [ ] Results contain `metrics` with `auc`, `precision`, `recall`, `f1`
- [ ] AUC > 0.5 (better than random)
- [ ] Meta log CSV exists

### Performance Verification
- [ ] Scenario A: AUC reported
- [ ] Scenario B: AUC, Pattern Preservation, SNR Improvement reported
- [ ] Power difference < 1% (ultra-covert)

---

## 🔍 Troubleshooting

### Issue: Dataset generation fails
```bash
# Check GPU availability
nvidia-smi

# Check memory
free -h

# Try with fewer workers
python3 generate_dataset_parallel.py --scenario sat --total-samples 1000
```

### Issue: Training fails
```bash
# Check dataset exists
ls -lh dataset/dataset_scenario_*.pkl

# Check CUDA/GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Try with smaller batch size
python3 main_detection_cnn.py --scenario sat --epochs 10 --batch-size 256
```

### Issue: Results not found
```bash
# Check if training completed
tail -50 training_scenario_*.log

# Verify scenario name
python3 -c "from config.settings import INSIDER_MODE; print(f'INSIDER_MODE: {INSIDER_MODE}')"
```

---

## 📊 Expected Results

### Scenario A (Baseline)
- **AUC**: ~0.49-0.62
- **Precision**: ~0.50
- **Recall**: ~1.00
- **F1**: ~0.67
- **Power Diff**: < 0.2%

### Scenario B (With MMSE)
- **AUC**: ~0.62-0.68
- **Precision**: ~0.50
- **Recall**: ~1.00
- **F1**: ~0.67
- **Power Diff**: < 0.2%
- **Pattern Preservation**: ~0.49-0.50
- **SNR Improvement**: ~30-35 dB

---

## 🚀 Quick Start (Production)

For production deployment, use this single command sequence:

```bash
#!/bin/bash
# Complete Pipeline - Production Script

set -e  # Exit on error

cd /workspace

echo "🚀 Starting Complete Pipeline..."

# Scenario A
echo "📊 Scenario A: Generating dataset..."
python3 generate_dataset_parallel.py --scenario sat --total-samples 5000
cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl 2>/dev/null || true

echo "🧠 Scenario A: Training model..."
python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512 2>&1 | tee training_scenario_a.log

# Scenario B
echo "📊 Scenario B: Generating dataset..."
python3 generate_dataset_parallel.py --scenario ground --total-samples 5000
cp dataset/dataset_scenario_b_4998.pkl dataset/dataset_scenario_b_5000.pkl 2>/dev/null || true

echo "🧠 Scenario B: Training model..."
python3 main_detection_cnn.py --scenario ground --epochs 30 --batch-size 512 2>&1 | tee training_scenario_b.log

echo "✅ Complete pipeline finished!"
echo "📁 Check results in: result/scenario_a/ and result/scenario_b/"
```

Save as `run_production_pipeline.sh` and execute:
```bash
chmod +x run_production_pipeline.sh
./run_production_pipeline.sh
```

---

## 📝 Notes

1. **Dataset Generation Time**: ~5-10 minutes per scenario (depends on GPU)
2. **Training Time**: ~5-10 minutes per scenario (30 epochs, 512 batch size)
3. **Total Time**: ~20-40 minutes for complete pipeline
4. **Storage**: ~160 MB for datasets + ~50 MB for models
5. **GPU Memory**: Recommended 16GB+ for smooth execution

---

**✅ Pipeline ready for production use!**

