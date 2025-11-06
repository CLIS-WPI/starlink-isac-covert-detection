#!/usr/bin/env python3
"""
Test if preprocessing preserves the pattern
"""
import pickle
import numpy as np

# Load dataset
d = pickle.load(open('dataset/dataset_samples200_sats12.pkl', 'rb'))
rx_grids = d['rx_grids']
labels = d['labels']

# Squeeze
X = np.squeeze(rx_grids, axis=1)  # (400, 10, 64)

# Separate classes
benign_X = X[labels == 0]
attack_X = X[labels == 1]

print("="*70)
print("BEFORE PREPROCESSING")
print("="*70)

# Raw magnitude
benign_mag = np.mean(np.abs(benign_X), axis=(0, 1))
attack_mag = np.mean(np.abs(attack_X), axis=(0, 1))
diff_before = (attack_mag[:16].mean() - benign_mag[:16].mean()) / benign_mag[:16].mean() * 100

print(f"Covert band (0-15) difference: {diff_before:.2f}%")

print("\n" + "="*70)
print("AFTER PREPROCESSING (like CNN does)")
print("="*70)

# Apply same preprocessing as CNN
magnitude = np.abs(X)  # (400, 10, 64)
phase = np.angle(X)

# Stack
X_proc = np.stack([magnitude, phase], axis=-1).astype(np.float32)

# GLOBAL normalization (NEW METHOD)
global_mag_max = np.max(X_proc[..., 0])
if global_mag_max > 0:
    X_proc[..., 0] /= global_mag_max

# Phase normalized
X_proc[..., 1] /= np.pi

# Now check pattern
benign_proc = X_proc[labels == 0][..., 0]  # magnitude channel
attack_proc = X_proc[labels == 1][..., 0]

benign_mag_proc = np.mean(benign_proc, axis=(0, 1))
attack_mag_proc = np.mean(attack_proc, axis=(0, 1))
diff_after = (attack_mag_proc[:16].mean() - benign_mag_proc[:16].mean()) / benign_mag_proc[:16].mean() * 100

print(f"Covert band (0-15) difference: {diff_after:.2f}%")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print(f"Pattern strength BEFORE preprocessing: {diff_before:.2f}%")
print(f"Pattern strength AFTER preprocessing:  {diff_after:.2f}%")
print(f"Pattern loss: {diff_before - diff_after:.2f}%")

if abs(diff_after) < 2.0:
    print("\n❌ PROBLEM: Per-sample normalization destroys pattern!")
    print("   → All samples normalized to [0,1], relative difference lost")
    print("   → CNN cannot learn without pattern!")
elif abs(diff_after) > 5.0:
    print("\n✅ Pattern preserved after preprocessing")
    print("   → CNN should be able to learn this")
else:
    print("\n⚠️  Pattern weak after preprocessing")
    print("   → Marginal, CNN may struggle")

print("="*70)
