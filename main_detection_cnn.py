#!/usr/bin/env python3
"""
🧠 CNN DETECTION PIPELINE
=========================
Main detection pipeline using CNN-based detector (with optional CSI fusion).

This replaces the RandomForest approach with deep learning for:
- Ultra-subtle covert channel detection (< 1% power difference)
- Automatic feature learning from raw OFDM grids
- Optional multi-modal fusion with CSI data

Usage:
    python3 main_detection_cnn.py --use-csi  # With CSI fusion
    python3 main_detection_cnn.py            # CNN-only
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# Configuration
from config.settings import (
    init_directories,
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA,
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR,
    SEED
)

# CNN Detector
from model.detector_cnn import CNNDetector


def main(use_csi=False, epochs=50, batch_size=32):
    """
    Main CNN detection pipeline.
    
    Args:
        use_csi: Whether to use CSI fusion (default: False)
        epochs: Number of training epochs
        batch_size: Training batch size
    
    Returns:
        success: Whether pipeline succeeded
        results: Dictionary with metrics and metadata
    """
    print("\n" + "#"*70)
    print("🧠 CNN COVERT CHANNEL DETECTION PIPELINE")
    print("#"*70)
    print()
    
    # Initialize directories
    init_directories()
    
    # Results dictionary
    results = {
        'success': False,
        'timestamp': time.time(),
        'config': {
            'detector': 'CNN' + ('+CSI' if use_csi else ''),
            'num_samples': NUM_SAMPLES_PER_CLASS * 2,
            'num_satellites': NUM_SATELLITES_FOR_TDOA,
            'epochs': epochs,
            'batch_size': batch_size,
            'seed': SEED
        }
    }
    
    # ===== Phase 1: Load Dataset =====
    print(f"\n{'='*70}")
    print("[Phase 1] Loading pre-generated dataset...")
    print(f"{'='*70}")
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        return False, results
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"  ✓ Loaded dataset from {dataset_path}")
    print(f"  → Total samples: {len(dataset['labels'])}")
    print(f"  → Benign: {np.sum(dataset['labels'] == 0)}")
    print(f"  → Attack: {np.sum(dataset['labels'] == 1)}")
    
    # Extract data
    X_grids = dataset['tx_grids']  # OFDM grids
    Y = dataset['labels']
    
    X_csi = None
    if use_csi and 'csi' in dataset:
        X_csi = dataset['csi']
        print(f"  ✓ CSI data available: shape {X_csi.shape}")
    elif use_csi:
        print(f"  ⚠️ CSI fusion requested but 'csi' not in dataset!")
        print(f"     Falling back to CNN-only mode.")
        use_csi = False
    
    # ===== Phase 2: Power Verification =====
    print(f"\n{'='*70}")
    print("[Phase 2] Power-preserving verification...")
    print(f"{'='*70}")
    
    benign_mask = (Y == 0)
    attack_mask = (Y == 1)
    
    benign_grids = np.squeeze(X_grids[benign_mask])
    attack_grids = np.squeeze(X_grids[attack_mask])
    
    benign_power = np.mean(np.abs(benign_grids) ** 2)
    attack_power = np.mean(np.abs(attack_grids) ** 2)
    power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    print(f"  ✓ Benign power: {benign_power:.6f}")
    print(f"  ✓ Attack power: {attack_power:.6f}")
    print(f"  ✓ Difference:   {power_diff_pct:.2f}%")
    
    if power_diff_pct < 5.0:
        print(f"  ✅ Ultra-covert: Power difference < 5% (truly stealthy!)")
    elif power_diff_pct < 10.0:
        print(f"  ✅ Covert: Power difference < 10%")
    else:
        print(f"  ⚠️ Warning: Power difference > 10% (may be detectable)")
    
    results['power_analysis'] = {
        'benign_power': float(benign_power),
        'attack_power': float(attack_power),
        'difference_pct': float(power_diff_pct)
    }
    
    # ===== Phase 3: Train/Test Split =====
    print(f"\n{'='*70}")
    print("[Phase 3] Splitting dataset...")
    print(f"{'='*70}")
    
    test_size = 0.3
    
    if use_csi:
        X_train, X_test, X_csi_train, X_csi_test, y_train, y_test = train_test_split(
            X_grids, X_csi, Y,
            test_size=test_size,
            stratify=Y,
            random_state=SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_grids, Y,
            test_size=test_size,
            stratify=Y,
            random_state=SEED
        )
        X_csi_train, X_csi_test = None, None
    
    print(f"  ✓ Training set: {len(y_train)} samples")
    print(f"  ✓ Test set:     {len(y_test)} samples")
    print(f"  ✓ Train benign: {np.sum(y_train == 0)}")
    print(f"  ✓ Train attack: {np.sum(y_train == 1)}")
    print(f"  ✓ Test benign:  {np.sum(y_test == 0)}")
    print(f"  ✓ Test attack:  {np.sum(y_test == 1)}")
    
    # ===== Phase 4: Train CNN Detector =====
    print(f"\n{'='*70}")
    print("[Phase 4] Training CNN detector...")
    print(f"{'='*70}")
    
    detector = CNNDetector(
        use_csi=use_csi,
        learning_rate=0.001,
        dropout_rate=0.3,
        random_state=SEED
    )
    
    # Train with validation split
    val_size = 0.2
    if use_csi:
        X_tr, X_val, X_csi_tr, X_csi_val, y_tr, y_val = train_test_split(
            X_train, X_csi_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=SEED
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=SEED
        )
        X_csi_tr, X_csi_val = None, None
    
    history = detector.train(
        X_tr, y_tr, X_csi_train=X_csi_tr,
        X_val=X_val, y_val=y_val, X_csi_val=X_csi_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # ===== Phase 5: Evaluate on Test Set =====
    print(f"\n{'='*70}")
    print("[Phase 5] Evaluating on test set...")
    print(f"{'='*70}")
    
    metrics = detector.evaluate(X_test, y_test, X_csi_test=X_csi_test)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  {'='*50}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  {'='*50}")
    
    if metrics['auc'] >= 0.95:
        print(f"  ✅ Excellent detection (AUC ≥ 0.95)")
    elif metrics['auc'] >= 0.85:
        print(f"  ✅ Good detection (AUC ≥ 0.85)")
    elif metrics['auc'] >= 0.70:
        print(f"  ⚠️ Moderate detection (AUC ≥ 0.70)")
    else:
        print(f"  ❌ Poor detection (AUC < 0.70)")
    
    results['metrics'] = metrics
    results['success'] = True
    
    # ===== Phase 6: Save Model =====
    print(f"\n{'='*70}")
    print("[Phase 6] Saving model...")
    print(f"{'='*70}")
    
    model_filename = f"cnn_detector{'_csi' if use_csi else ''}.keras"
    model_path = f"{MODEL_DIR}/{model_filename}"
    detector.save(model_path)
    
    results['model_path'] = model_path
    
    # ===== Save Results =====
    result_path = f"{RESULT_DIR}/detection_results_cnn.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {result_path}")
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    print("📋 PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Dataset:        {NUM_SAMPLES_PER_CLASS * 2} samples")
    print(f"  Detector:       CNN{'+CSI' if use_csi else ''}")
    print(f"  Power diff:     {power_diff_pct:.2f}%")
    print(f"  Test AUC:       {metrics['auc']:.4f}")
    print(f"  Model saved:    {model_path}")
    print(f"{'='*70}")
    
    return True, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN-based covert channel detection')
    parser.add_argument('--use-csi', action='store_true',
                       help='Enable CSI fusion (multi-modal)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    
    args = parser.parse_args()
    
    try:
        success, results = main(
            use_csi=args.use_csi,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if success:
            print("\n✅ CNN detection pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ CNN detection pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
