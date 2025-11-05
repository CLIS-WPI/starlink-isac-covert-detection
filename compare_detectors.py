#!/usr/bin/env python3
"""
⚖️ DETECTOR COMPARISON: RandomForest vs CNN
===========================================
Run both detectors on the same dataset and compare performance.

Usage:
    python3 compare_detectors.py
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
from config.settings import (
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA,
    DATASET_DIR,
    RESULT_DIR,
    SEED
)

# Detectors
from model.detector_frequency import FrequencyDetector
from model.detector_cnn import CNNDetector


def main():
    """Compare RandomForest and CNN detectors."""
    
    print("\n" + "="*70)
    print("⚖️  DETECTOR COMPARISON: RandomForest vs CNN")
    print("="*70)
    
    # Load dataset
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"\n📂 Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    print(f"  ✓ Total samples: {len(Y)}")
    print(f"  ✓ Benign: {np.sum(Y == 0)}")
    print(f"  ✓ Attack: {np.sum(Y == 1)}")
    
    # Power analysis
    benign_mask = (Y == 0)
    attack_mask = (Y == 1)
    benign_power = np.mean(np.abs(np.squeeze(X_grids[benign_mask])) ** 2)
    attack_power = np.mean(np.abs(np.squeeze(X_grids[attack_mask])) ** 2)
    power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    print(f"\n⚡ Power Analysis:")
    print(f"  Benign: {benign_power:.6f}")
    print(f"  Attack: {attack_power:.6f}")
    print(f"  Diff:   {power_diff_pct:.2f}%")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_grids, Y,
        test_size=0.3,
        stratify=Y,
        random_state=SEED
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"  Train: {len(y_train)} samples")
    print(f"  Test:  {len(y_test)} samples")
    
    results = {
        'power_diff_pct': float(power_diff_pct),
        'num_samples': len(Y),
        'detectors': {}
    }
    
    # ===== Detector 1: RandomForest =====
    print(f"\n{'='*70}")
    print("🌲 Testing RandomForest Detector")
    print(f"{'='*70}")
    
    rf_start = time.time()
    
    rf_detector = FrequencyDetector(
        n_estimators=100,
        max_depth=12,
        random_state=SEED,
        n_jobs=-1,
        mask_weight=10.0
    )
    
    rf_detector.train(X_train, y_train)
    rf_metrics = rf_detector.evaluate(X_test, y_test)
    
    rf_time = time.time() - rf_start
    
    print(f"\n📊 RandomForest Results:")
    print(f"  AUC:       {rf_metrics['auc']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall:    {rf_metrics['recall']:.4f}")
    print(f"  F1:        {rf_metrics['f1']:.4f}")
    print(f"  Time:      {rf_time:.2f}s")
    
    results['detectors']['RandomForest'] = {
        'metrics': rf_metrics,
        'time_seconds': rf_time
    }
    
    # ===== Detector 2: CNN =====
    print(f"\n{'='*70}")
    print("🧠 Testing CNN Detector")
    print(f"{'='*70}")
    
    cnn_start = time.time()
    
    cnn_detector = CNNDetector(
        use_csi=False,
        learning_rate=0.001,
        dropout_rate=0.3,
        random_state=SEED
    )
    
    # Further split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=SEED
    )
    
    cnn_detector.train(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        epochs=50,
        batch_size=32,
        verbose=0  # Suppress training logs
    )
    
    cnn_metrics = cnn_detector.evaluate(X_test, y_test)
    
    cnn_time = time.time() - cnn_start
    
    print(f"\n📊 CNN Results:")
    print(f"  AUC:       {cnn_metrics['auc']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1:        {cnn_metrics['f1']:.4f}")
    print(f"  Time:      {cnn_time:.2f}s")
    
    results['detectors']['CNN'] = {
        'metrics': cnn_metrics,
        'time_seconds': cnn_time
    }
    
    # ===== Comparison =====
    print(f"\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<15} {'RandomForest':<15} {'CNN':<15} {'Winner':<15}")
    print("-" * 60)
    
    for metric in ['auc', 'precision', 'recall', 'f1']:
        rf_val = rf_metrics[metric]
        cnn_val = cnn_metrics[metric]
        
        winner = "CNN" if cnn_val > rf_val else "RF" if rf_val > cnn_val else "Tie"
        winner_symbol = "🧠" if winner == "CNN" else "🌲" if winner == "RF" else "🤝"
        
        print(f"{metric.upper():<15} {rf_val:<15.4f} {cnn_val:<15.4f} {winner_symbol} {winner:<15}")
    
    print("-" * 60)
    print(f"{'Time (s)':<15} {rf_time:<15.2f} {cnn_time:<15.2f}")
    
    # Improvement analysis
    auc_improvement = ((cnn_metrics['auc'] - rf_metrics['auc']) / rf_metrics['auc']) * 100
    
    print(f"\n💡 Analysis:")
    print(f"  Power difference: {power_diff_pct:.2f}%")
    
    if power_diff_pct < 1.0:
        print(f"  → Ultra-covert (< 1% power diff)")
    elif power_diff_pct < 5.0:
        print(f"  → Covert (< 5% power diff)")
    else:
        print(f"  → Detectable (> 5% power diff)")
    
    print(f"\n  CNN AUC improvement: {auc_improvement:+.1f}%")
    
    if auc_improvement > 20:
        print(f"  ✅ CNN significantly outperforms RF (as expected for covert channels)")
    elif auc_improvement > 5:
        print(f"  ✅ CNN moderately outperforms RF")
    elif auc_improvement > -5:
        print(f"  🤝 Similar performance (both work or both struggle)")
    else:
        print(f"  ⚠️ RF outperforms CNN (unexpected - check CNN training)")
    
    if cnn_metrics['auc'] > 0.85 and power_diff_pct < 5.0:
        print(f"\n  🎯 Success! CNN detects ultra-subtle covert channel (AUC > 0.85)")
    elif cnn_metrics['auc'] > 0.75 and power_diff_pct < 1.0:
        print(f"\n  ✅ Good! CNN works on ultra-covert channel (< 1% power diff)")
    elif rf_metrics['auc'] < 0.65 and cnn_metrics['auc'] < 0.65:
        print(f"\n  ⚠️ Both detectors struggle. Consider:")
        print(f"     - Increasing COVERT_AMP")
        print(f"     - Collecting more data")
        print(f"     - Checking mask alignment")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results = convert_to_native(results)
    
    # Save results
    result_path = f"{RESULT_DIR}/detector_comparison.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {result_path}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
