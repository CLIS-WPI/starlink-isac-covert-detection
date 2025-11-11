#!/bin/bash
# =====================================
# Scenario A: Complete Pipeline
# =====================================
# Generates dataset, trains model, and produces results for Scenario A
# Scenario A: Single-hop Downlink (Insider@Satellite)

set -e  # Exit on error

cd /workspace

echo "="*80
echo "🚀 SCENARIO A: Complete Pipeline"
echo "="*80
echo "Scenario: Single-hop Downlink (Insider@Satellite)"
echo "Started at: $(date)"
echo ""

# ===== Step 1: Generate Dataset =====
echo "="*80
echo "📊 Step 1: Generating Dataset"
echo "="*80

python3 generate_dataset_parallel.py \
    --scenario sat \
    --total-samples 5000

# Verify dataset
if [ -f "dataset/dataset_scenario_a_5000.pkl" ]; then
    echo "✅ Dataset created: dataset_scenario_a_5000.pkl"
    # Clean up old file if exists
    if [ -f "dataset/dataset_scenario_a_4998.pkl" ]; then
        rm dataset/dataset_scenario_a_4998.pkl
        echo "✅ Removed old file: dataset_scenario_a_4998.pkl"
    fi
elif [ -f "dataset/dataset_scenario_a_4998.pkl" ]; then
    cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl
    rm dataset/dataset_scenario_a_4998.pkl
    echo "✅ Dataset created: dataset_scenario_a_5000.pkl (from 4998, old file removed)"
else
    echo "❌ Dataset generation failed!"
    exit 1
fi

# Verify dataset contents
echo ""
echo "🔍 Verifying dataset..."
python3 << 'EOF'
import pickle
from pathlib import Path

dataset_file = Path('dataset/dataset_scenario_a_5000.pkl')
if dataset_file.exists():
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"   ✅ Samples: {len(dataset['labels'])}")
    print(f"   ✅ Size: {dataset_file.stat().st_size / (1024**2):.2f} MB")
    print(f"   ✅ TX shape: {dataset['tx_grids'].shape}")
    print(f"   ✅ RX shape: {dataset['rx_grids'].shape}")
    print(f"   ✅ Benign: {sum(dataset['labels'] == 0)}")
    print(f"   ✅ Attack: {sum(dataset['labels'] == 1)}")
else:
    print("   ❌ Dataset file not found!")
    exit(1)
EOF

# ===== Step 2: Train CNN Model =====
echo ""
echo "="*80
echo "🧠 Step 2: Training CNN Model"
echo "="*80

python3 main_detection_cnn.py \
    --scenario sat \
    --epochs 30 \
    --batch-size 512 \
    2>&1 | tee training_scenario_a.log

# Verify training completed
if [ ! -f "result/scenario_a/detection_results_cnn.json" ]; then
    echo "❌ Training failed - results file not found!"
    exit 1
fi

echo ""
echo "✅ Training completed!"

# ===== Step 3: Generate Results Report =====
echo ""
echo "="*80
echo "📊 Step 3: Generating Results Report"
echo "="*80

python3 << 'EOF'
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

print("\n" + "="*80)
print("📊 SCENARIO A - FINAL RESULTS")
print("="*80)

# Load dataset
dataset_file = Path('dataset/dataset_scenario_a_5000.pkl')
with open(dataset_file, 'rb') as f:
    dataset = pickle.load(f)

# Load results
result_file = Path('result/scenario_a/detection_results_cnn.json')
with open(result_file, 'r') as f:
    results = json.load(f)

metrics = results.get('metrics', {})
power_analysis = results.get('power_analysis', {})

print(f"\n✅ Dataset Information:")
print(f"   • Total samples: {len(dataset['labels'])}")
print(f"   • Dataset size: {dataset_file.stat().st_size / (1024**2):.2f} MB")
print(f"   • Benign samples: {sum(dataset['labels'] == 0)}")
print(f"   • Attack samples: {sum(dataset['labels'] == 1)}")

print(f"\n📊 Detection Performance:")
print(f"   • AUC: {metrics.get('auc', 0):.4f}")
print(f"   • Precision: {metrics.get('precision', 0):.4f}")
print(f"   • Recall: {metrics.get('recall', 0):.4f}")
print(f"   • F1 Score: {metrics.get('f1', 0):.4f}")
print(f"   • Optimal Threshold: {metrics.get('threshold', 0):.4f}")

print(f"\n📊 Power Analysis:")
power_diff = power_analysis.get('difference_pct', 0)
print(f"   • Power Difference: {power_diff:.4f}%")
print(f"   • Status: {'✅ Ultra-covert' if power_diff < 0.2 else '⚠️  Visible' if power_diff < 1.0 else '❌ Detectable'}")

print(f"\n📁 Output Files:")
print(f"   • Model: model/scenario_a/cnn_detector.keras")
print(f"   • Results: result/scenario_a/detection_results_cnn.json")
print(f"   • Normalization: model/scenario_a/cnn_detector_norm.pkl")
print(f"   • Training log: training_scenario_a.log")

print("\n" + "="*80)
print("✅ SCENARIO A PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)
EOF

echo ""
echo "Finished at: $(date)"
echo "="*80

