#!/usr/bin/env python3
"""
🔍 Dataset Balance & Reproducibility Check
==========================================
Verify class balance and random_state consistency
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def check_dataset_balance(dataset_path):
    """Check class balance in dataset"""
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("\n" + "="*70)
    print("🔍 DATASET BALANCE & REPRODUCIBILITY CHECK")
    print("="*70)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X = dataset['tx_grids']
    Y = dataset['labels']
    
    # Overall balance
    unique, counts = np.unique(Y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\n📊 Overall Dataset:")
    print(f"  Total samples: {len(Y)}")
    print(f"  Class 0 (benign): {class_dist.get(0, 0)} samples ({class_dist.get(0, 0)/len(Y)*100:.1f}%)")
    print(f"  Class 1 (attack): {class_dist.get(1, 0)} samples ({class_dist.get(1, 0)/len(Y)*100:.1f}%)")
    
    if len(class_dist) == 2:
        imbalance_ratio = max(counts) / min(counts)
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print(f"  ⚠️  IMBALANCED - Consider class weights!")
            # Calculate suggested weights
            total = sum(counts)
            weights = {i: total / (len(counts) * count) for i, count in zip(unique, counts)}
            print(f"  💡 Suggested class_weight: {weights}")
        else:
            print(f"  ✅ Well balanced")
            print(f"  💡 Recommended class_weight: {{0: 1.0, 1: 1.0}}")
    
    # Test reproducibility with fixed seed
    print(f"\n🔁 Reproducibility Test (SEED=42):")
    
    SEED = 42
    test_size = 0.3
    
    # Split 1
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=SEED
    )
    
    # Split 2 (same seed)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=SEED
    )
    
    # Check if identical
    train_match = np.array_equal(y_train1, y_train2)
    test_match = np.array_equal(y_test1, y_test2)
    
    print(f"  Train set matches: {train_match} ✅" if train_match else f"  Train set matches: {train_match} ❌")
    print(f"  Test set matches:  {test_match} ✅" if test_match else f"  Test set matches:  {test_match} ❌")
    
    if train_match and test_match:
        print(f"  ✅ Splits are reproducible with SEED={SEED}")
    else:
        print(f"  ❌ Splits NOT reproducible - check random_state!")
    
    # Check train/test balance
    print(f"\n📊 Train/Test Split Balance (test_size={test_size}):")
    
    train_unique, train_counts = np.unique(y_train1, return_counts=True)
    test_unique, test_counts = np.unique(y_test1, return_counts=True)
    
    train_dist = dict(zip(train_unique, train_counts))
    test_dist = dict(zip(test_unique, test_counts))
    
    print(f"  Training set:")
    print(f"    Class 0: {train_dist.get(0, 0)} samples ({train_dist.get(0, 0)/len(y_train1)*100:.1f}%)")
    print(f"    Class 1: {train_dist.get(1, 0)} samples ({train_dist.get(1, 0)/len(y_train1)*100:.1f}%)")
    
    print(f"  Test set:")
    print(f"    Class 0: {test_dist.get(0, 0)} samples ({test_dist.get(0, 0)/len(y_test1)*100:.1f}%)")
    print(f"    Class 1: {test_dist.get(1, 0)} samples ({test_dist.get(1, 0)/len(y_test1)*100:.1f}%)")
    
    # Check if stratify worked
    train_ratio = train_dist.get(1, 0) / (train_dist.get(0, 1) + train_dist.get(1, 0))
    test_ratio = test_dist.get(1, 0) / (test_dist.get(0, 1) + test_dist.get(1, 0))
    overall_ratio = class_dist.get(1, 0) / len(Y)
    
    ratio_diff = abs(train_ratio - test_ratio)
    
    if ratio_diff < 0.02:  # Within 2%
        print(f"  ✅ Stratification successful (train={train_ratio:.2%}, test={test_ratio:.2%})")
    else:
        print(f"  ⚠️  Stratification issue (train={train_ratio:.2%}, test={test_ratio:.2%})")
    
    # Summary
    print(f"\n" + "="*70)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print(f"\n✅ Checklist:")
    print(f"  {'[✓]' if imbalance_ratio <= 1.5 else '[✗]'} Class balance is good (ratio ≤ 1.5:1)")
    print(f"  {'[✓]' if train_match and test_match else '[✗]'} Random splits are reproducible")
    print(f"  {'[✓]' if ratio_diff < 0.02 else '[✗]'} Stratification preserves class distribution")
    
    print(f"\n💡 Recommendations:")
    if imbalance_ratio <= 1.5:
        print(f"  - Use default class_weight={{0: 1.0, 1: 1.0}}")
    else:
        weights = {i: total / (len(counts) * count) for i, count in zip(unique, counts)}
        print(f"  - Use class_weight={weights}")
    
    print(f"  - Keep SEED=42 for all experiments")
    print(f"  - Always use stratify=Y in train_test_split")
    print(f"  - Document any changes to SEED in results")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys
    from config.settings import NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA, DATASET_DIR
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    check_dataset_balance(dataset_path)
