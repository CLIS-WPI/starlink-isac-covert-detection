#!/usr/bin/env python3
"""
🧪 Generalization Test Script
==============================
Test model generalization on:
1. Different injection patterns (different subcarriers)
2. Different power levels (COVERT_AMP)
3. Different channel conditions (SNR)
"""

import os
import sys
import pickle
import numpy as np
import argparse
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from config.settings import *
from model.detector_cnn import CNNDetector


def test_generalization(model_path, test_datasets, use_csi=False):
    """
    Test model generalization on different test datasets.
    
    Args:
        model_path: Path to saved model (.keras file)
        test_datasets: List of (dataset_path, description) tuples
        use_csi: Whether model uses CSI fusion
    """
    print("="*70)
    print("🧪 GENERALIZATION TEST")
    print("="*70)
    
    # Load trained model
    print(f"\n📂 Loading model from: {model_path}")
    detector = CNNDetector(use_csi=use_csi)
    detector.model = tf.keras.models.load_model(model_path)
    detector.is_trained = True
    
    # Load normalization stats from training (if available)
    # Note: In real scenario, these should be saved with the model
    # For now, we'll recompute from first test dataset (not ideal but works)
    
    results = []
    
    for dataset_path, description in test_datasets:
        print(f"\n{'='*70}")
        print(f"📊 Testing on: {description}")
        print(f"   Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*70}")
        
        # Load test dataset
        with open(dataset_path, 'rb') as f:
            test_data = pickle.load(f)
        
        labels = np.array(test_data['labels'])
        tx_grids = test_data['tx_grids']
        X_csi_test = None
        if use_csi and 'csi_est' in test_data:
            X_csi_test = test_data['csi_est']
            if X_csi_test.ndim == 4 and X_csi_test.shape[1] == 1:
                X_csi_test = np.squeeze(X_csi_test, axis=1)
        
        # Evaluate
        metrics = detector.evaluate(tx_grids, labels, X_csi_test)
        
        print(f"\n  Results:")
        print(f"    AUC:       {metrics['auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        
        results.append({
            'description': description,
            'dataset': os.path.basename(dataset_path),
            'metrics': metrics
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    
    for r in results:
        print(f"\n  {r['description']}:")
        print(f"    AUC: {r['metrics']['auc']:.4f}")
    
    # Calculate average AUC
    avg_auc = np.mean([r['metrics']['auc'] for r in results])
    print(f"\n  Average AUC: {avg_auc:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test model generalization")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.keras)")
    parser.add_argument("--use-csi", action="store_true", help="Model uses CSI fusion")
    parser.add_argument("--test-datasets", nargs="+", help="List of test dataset paths")
    parser.add_argument("--test-descriptions", nargs="+", help="List of descriptions for test datasets")
    
    args = parser.parse_args()
    
    if args.test_datasets and args.test_descriptions:
        if len(args.test_datasets) != len(args.test_descriptions):
            print("❌ Error: Number of datasets and descriptions must match!")
            sys.exit(1)
        test_datasets = list(zip(args.test_datasets, args.test_descriptions))
    else:
        # Default: test on same dataset (baseline)
        default_dataset = os.path.join(DATASET_DIR, "dataset_samples500_sats12.pkl")
        if os.path.exists(default_dataset):
            test_datasets = [(default_dataset, "Baseline (same as training)")]
        else:
            print("❌ Error: No test datasets provided and default dataset not found!")
            sys.exit(1)
    
    test_generalization(args.model, test_datasets, args.use_csi)


if __name__ == "__main__":
    main()

