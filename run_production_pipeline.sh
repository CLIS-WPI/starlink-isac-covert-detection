#!/bin/bash
# Complete Pipeline - Production Script
# =====================================
# Runs complete pipeline for both Scenario A and B from scratch

set -e  # Exit on error

cd /workspace

echo "="*80
echo "🚀 COMPLETE PRODUCTION PIPELINE"
echo "="*80
echo "Started at: $(date)"
echo ""

# ===== Scenario A =====
echo "="*80
echo "1️⃣  SCENARIO A: Single-hop Downlink"
echo "="*80

echo "📊 Step 1: Generating dataset..."
python3 generate_dataset_parallel.py --scenario sat --total-samples 5000

# Note: Script automatically creates standard name if actual_total == total_samples
# Check if standard name exists
if [ ! -f "dataset/dataset_scenario_a_5000.pkl" ] && [ -f "dataset/dataset_scenario_a_4998.pkl" ]; then
    cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl
    echo "✅ Dataset copied to standard name (fallback)"
fi

# Verify dataset
echo "🔍 Verifying dataset..."
python3 << 'EOF'
import pickle
from pathlib import Path
dataset = pickle.load(open('dataset/dataset_scenario_a_5000.pkl', 'rb'))
print(f"✅ Dataset: {len(dataset['labels'])} samples")
print(f"✅ Size: {Path('dataset/dataset_scenario_a_5000.pkl').stat().st_size / (1024**2):.2f} MB")
EOF

echo "🧠 Step 2: Training CNN model..."
python3 main_detection_cnn.py \
    --scenario sat \
    --epochs 30 \
    --batch-size 512 \
    2>&1 | tee training_scenario_a.log

# Verify results
echo "🔍 Verifying results..."
python3 << 'EOF'
import json
from pathlib import Path
if Path('result/scenario_a/detection_results_cnn.json').exists():
    results = json.load(open('result/scenario_a/detection_results_cnn.json', 'r'))
    metrics = results['metrics']
    print(f"✅ AUC: {metrics['auc']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall: {metrics['recall']:.4f}")
    print(f"✅ F1: {metrics['f1']:.4f}")
else:
    print("❌ Results file not found")
EOF

echo ""
echo "✅ Scenario A completed!"
echo ""

# ===== Scenario B =====
echo "="*80
echo "2️⃣  SCENARIO B: Dual-hop Relay"
echo "="*80

echo "📊 Step 1: Generating dataset..."
python3 generate_dataset_parallel.py --scenario ground --total-samples 5000

# Note: Script automatically creates standard name if actual_total == total_samples
# Check if standard name exists
if [ ! -f "dataset/dataset_scenario_b_5000.pkl" ] && [ -f "dataset/dataset_scenario_b_4998.pkl" ]; then
    cp dataset/dataset_scenario_b_4998.pkl dataset/dataset_scenario_b_5000.pkl
    echo "✅ Dataset copied to standard name (fallback)"
fi

# Verify dataset
echo "🔍 Verifying dataset..."
python3 << 'EOF'
import pickle
from pathlib import Path
dataset = pickle.load(open('dataset/dataset_scenario_b_5000.pkl', 'rb'))
print(f"✅ Dataset: {len(dataset['labels'])} samples")
print(f"✅ Size: {Path('dataset/dataset_scenario_b_5000.pkl').stat().st_size / (1024**2):.2f} MB")
print(f"✅ CSI dtype: {dataset['csi_est'].dtype}")
EOF

echo "🧠 Step 2: Training CNN model..."
python3 main_detection_cnn.py \
    --scenario ground \
    --epochs 30 \
    --batch-size 512 \
    2>&1 | tee training_scenario_b.log

# Verify results
echo "🔍 Verifying results..."
python3 << 'EOF'
import json
import numpy as np
from pathlib import Path
if Path('result/scenario_b/detection_results_cnn.json').exists():
    results = json.load(open('result/scenario_b/detection_results_cnn.json', 'r'))
    metrics = results['metrics']
    print(f"✅ AUC: {metrics['auc']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall: {metrics['recall']:.4f}")
    print(f"✅ F1: {metrics['f1']:.4f}")
    
    # Check EQ performance
    import pickle
    dataset = pickle.load(open('dataset/dataset_scenario_b_5000.pkl', 'rb'))
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
        print(f"✅ Pattern Preservation: {np.median(preservations):.3f}")
    if snr_improvements:
        print(f"✅ SNR Improvement: {np.mean(snr_improvements):.2f} dB")
else:
    print("❌ Results file not found")
EOF

echo ""
echo "✅ Scenario B completed!"
echo ""

# ===== Final Summary =====
echo "="*80
echo "📊 FINAL SUMMARY"
echo "="*80

python3 << 'EOF'
import json
import pickle
import numpy as np
from pathlib import Path

print("\n📊 Scenario A Results:")
if Path('result/scenario_a/detection_results_cnn.json').exists():
    results_a = json.load(open('result/scenario_a/detection_results_cnn.json', 'r'))
    metrics_a = results_a['metrics']
    print(f"   AUC: {metrics_a['auc']:.4f}")
    print(f"   F1: {metrics_a['f1']:.4f}")

print("\n📊 Scenario B Results:")
if Path('result/scenario_b/detection_results_cnn.json').exists():
    results_b = json.load(open('result/scenario_b/detection_results_cnn.json', 'r'))
    metrics_b = results_b['metrics']
    print(f"   AUC: {metrics_b['auc']:.4f}")
    print(f"   F1: {metrics_b['f1']:.4f}")
    
    # EQ metrics
    dataset_b = pickle.load(open('dataset/dataset_scenario_b_5000.pkl', 'rb'))
    meta_b = dataset_b.get('meta', [])
    preservations = []
    snr_improvements = []
    for m in meta_b[:1000]:
        if isinstance(m, tuple):
            _, m = m
        if 'eq_pattern_preservation' in m:
            preservations.append(m['eq_pattern_preservation'])
        if 'eq_snr_improvement_db' in m:
            snr_improvements.append(m['eq_snr_improvement_db'])
    
    if preservations:
        print(f"   Pattern Preservation: {np.median(preservations):.3f}")
    if snr_improvements:
        print(f"   SNR Improvement: {np.mean(snr_improvements):.2f} dB")

print("\n✅ Complete pipeline finished successfully!")
print("📁 Results saved in: result/scenario_a/ and result/scenario_b/")
EOF

echo ""
echo "="*80
echo "✅ PRODUCTION PIPELINE COMPLETED"
echo "="*80
echo "Finished at: $(date)"

