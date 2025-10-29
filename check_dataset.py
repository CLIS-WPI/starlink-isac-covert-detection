#!/usr/bin/env python3
"""Quick dataset sanity check"""
import pickle
import numpy as np

print("Loading dataset...")
with open("dataset/dataset_samples1500_sats12.pkl", "rb") as f:
    dataset = pickle.load(f)

print("\n=== DATASET SANITY CHECK ===")

# Check for NaN/Inf
for key in ['iq_samples', 'csi', 'radar_echo']:
    if key in dataset:
        data = np.array(dataset[key])
        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))
        print(f"{key:15} NaN: {has_nan:5}  Inf: {has_inf:5}  Shape: {data.shape}")
        
        if has_nan or has_inf:
            print(f"  ❌ PROBLEM in {key}!")
            # Find which samples
            nan_samples = np.where(np.any(np.isnan(data.reshape(len(data), -1)), axis=1))[0]
            inf_samples = np.where(np.any(np.isinf(data.reshape(len(data), -1)), axis=1))[0]
            if len(nan_samples) > 0:
                print(f"     NaN in samples: {nan_samples[:10]}...")
            if len(inf_samples) > 0:
                print(f"     Inf in samples: {inf_samples[:10]}...")

# Check labels
labels = np.array(dataset['labels'])
print(f"\nLabels: {np.unique(labels, return_counts=True)}")

# Check power
print("\n=== POWER CHECK ===")
benign_idx = np.where(labels == 0)[0][:100]
attack_idx = np.where(labels == 1)[0][:100]

benign_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in benign_idx])
attack_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in attack_idx])

print(f"Benign power: {benign_power:.6e}")
print(f"Attack power: {attack_power:.6e}")
print(f"Ratio: {attack_power / benign_power:.4f}")

print("\n✓ Check complete")
