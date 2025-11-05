#!/usr/bin/env python3
"""
🔍 CNN Diagnostic Script
========================
Diagnose why CNN is struggling with detection.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

print("="*70)
print("🔍 CNN DIAGNOSTIC ANALYSIS")
print("="*70)

# Load dataset
dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found: {dataset_path}")
    sys.exit(1)

print(f"\n📂 Loading {dataset_path}...")
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

X_grids = dataset['tx_grids']
Y = dataset['labels']

print(f"  ✓ Loaded {len(Y)} samples")
print(f"  ✓ tx_grids shape: {X_grids.shape}")

# Separate benign and attack
benign_mask = (Y == 0)
attack_mask = (Y == 1)

benign_grids = np.squeeze(X_grids[benign_mask])
attack_grids = np.squeeze(X_grids[attack_mask])

print(f"\n📊 Data Shapes:")
print(f"  Benign grids: {benign_grids.shape}")
print(f"  Attack grids: {attack_grids.shape}")

# Power analysis
benign_power = np.mean(np.abs(benign_grids) ** 2, axis=(1, 2))
attack_power = np.mean(np.abs(attack_grids) ** 2, axis=(1, 2))

print(f"\n⚡ Power Distribution:")
print(f"  Benign: mean={np.mean(benign_power):.4f}, std={np.std(benign_power):.4f}")
print(f"  Attack: mean={np.mean(attack_power):.4f}, std={np.std(attack_power):.4f}")
print(f"  Diff:   {abs(np.mean(attack_power) - np.mean(benign_power)) / np.mean(benign_power) * 100:.2f}%")

# Check if distributions overlap
overlap = (np.min(attack_power) < np.max(benign_power) and 
           np.max(attack_power) > np.min(benign_power))
print(f"  Overlap: {'YES ⚠️ (hard to separate)' if overlap else 'NO ✅ (easy to separate)'}")

# Magnitude analysis in injection region
print(f"\n🎯 Injection Region Analysis:")
print(f"  Expected injection: symbols [1:8], subcarriers [0:32]")

# Extract injection region
benign_injection = benign_grids[:, 1:8, 0:32]  # (N, 7, 32)
attack_injection = attack_grids[:, 1:8, 0:32]

benign_mag_inj = np.abs(benign_injection)
attack_mag_inj = np.abs(attack_injection)

print(f"  Benign magnitude (injection region): {np.mean(benign_mag_inj):.4f} ± {np.std(benign_mag_inj):.4f}")
print(f"  Attack magnitude (injection region): {np.mean(attack_mag_inj):.4f} ± {np.std(attack_mag_inj):.4f}")
print(f"  Difference: {abs(np.mean(attack_mag_inj) - np.mean(benign_mag_inj)) / np.mean(benign_mag_inj) * 100:.2f}%")

# Statistical separability (t-test)
from scipy import stats
t_stat, p_value = stats.ttest_ind(benign_mag_inj.flatten(), attack_mag_inj.flatten())
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
if p_value < 0.01:
    print(f"  ✅ Statistically significant difference (p < 0.01)")
elif p_value < 0.05:
    print(f"  ⚠️ Marginally significant (p < 0.05)")
else:
    print(f"  ❌ NOT statistically significant (p >= 0.05)")

# Visualize average grids
print(f"\n📊 Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Average magnitude
avg_benign_mag = np.mean(np.abs(benign_grids), axis=0)
avg_attack_mag = np.mean(np.abs(attack_grids), axis=0)
diff_mag = avg_attack_mag - avg_benign_mag

im1 = axes[0, 0].imshow(avg_benign_mag, aspect='auto', cmap='viridis')
axes[0, 0].set_title('Average Benign Magnitude')
axes[0, 0].set_xlabel('Subcarrier')
axes[0, 0].set_ylabel('Symbol')
plt.colorbar(im1, ax=axes[0, 0])

# Mark injection region
axes[0, 0].axhspan(1, 8, xmin=0, xmax=32/64, alpha=0.2, color='red', label='Injection')

im2 = axes[0, 1].imshow(avg_attack_mag, aspect='auto', cmap='viridis')
axes[0, 1].set_title('Average Attack Magnitude')
axes[0, 1].set_xlabel('Subcarrier')
axes[0, 1].set_ylabel('Symbol')
plt.colorbar(im2, ax=axes[0, 1])
axes[0, 1].axhspan(1, 8, xmin=0, xmax=32/64, alpha=0.2, color='red')

im3 = axes[0, 2].imshow(diff_mag, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
axes[0, 2].set_title('Difference (Attack - Benign)')
axes[0, 2].set_xlabel('Subcarrier')
axes[0, 2].set_ylabel('Symbol')
plt.colorbar(im3, ax=axes[0, 2])
axes[0, 2].axhspan(1, 8, xmin=0, xmax=32/64, alpha=0.2, color='green')

# Power distributions
axes[1, 0].hist(benign_power, bins=30, alpha=0.5, label='Benign', color='blue')
axes[1, 0].hist(attack_power, bins=30, alpha=0.5, label='Attack', color='red')
axes[1, 0].set_xlabel('Power')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Power Distribution')
axes[1, 0].legend()

# Magnitude per symbol (in injection region)
benign_per_symbol = np.mean(benign_mag_inj, axis=(0, 2))  # (7,)
attack_per_symbol = np.mean(attack_mag_inj, axis=(0, 2))

axes[1, 1].plot(range(1, 8), benign_per_symbol, 'o-', label='Benign', color='blue')
axes[1, 1].plot(range(1, 8), attack_per_symbol, 'o-', label='Attack', color='red')
axes[1, 1].set_xlabel('Symbol Index')
axes[1, 1].set_ylabel('Average Magnitude')
axes[1, 1].set_title('Per-Symbol Magnitude (Injection Region)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Magnitude per subcarrier (in injection region)
benign_per_sc = np.mean(benign_mag_inj, axis=(0, 1))  # (32,)
attack_per_sc = np.mean(attack_mag_inj, axis=(0, 1))

axes[1, 2].plot(range(32), benign_per_sc, '-', label='Benign', color='blue', alpha=0.7)
axes[1, 2].plot(range(32), attack_per_sc, '-', label='Attack', color='red', alpha=0.7)
axes[1, 2].set_xlabel('Subcarrier Index')
axes[1, 2].set_ylabel('Average Magnitude')
axes[1, 2].set_title('Per-Subcarrier Magnitude (Injection Region)')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plot_path = 'result/cnn_diagnostic.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved visualization to {plot_path}")

# Recommendations
print(f"\n💡 RECOMMENDATIONS:")
print("="*70)

if p_value > 0.05:
    print("❌ PROBLEM: Signal difference NOT statistically significant")
    print("   → Injection is too weak or not in expected region")
    print("   → SOLUTION: Increase COVERT_AMP to 0.70-0.80")

if abs(np.mean(attack_power) - np.mean(benign_power)) / np.mean(benign_power) < 0.03:
    print("❌ PROBLEM: Power difference < 3% (very subtle)")
    print("   → CNN needs more data or stronger signal")
    print("   → SOLUTION: Increase NUM_SAMPLES_PER_CLASS to 500+")

if len(Y) < 400:
    print("⚠️ WARNING: Dataset too small for CNN (< 400 samples)")
    print("   → CNNs need more data than RandomForest")
    print("   → SOLUTION: NUM_SAMPLES_PER_CLASS = 200-500")

print("\n✅ RECOMMENDED SETTINGS:")
print("  COVERT_AMP = 0.70  # Stronger signal for CNN")
print("  NUM_SAMPLES_PER_CLASS = 300  # More data")
print("  ADD_NOISE = False  # Keep disabled during testing")

print("="*70)
