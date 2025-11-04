#!/usr/bin/env python3
"""
🔍 DEBUG Script - مورد 3 & 9
بررسی تفاوت طیفی و energy ratio بین benign و attack
"""

import pickle
import numpy as np
import sys

# Load dataset
dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset_samples100_sats12.pkl"

print(f"📂 Loading {dataset_path}...")
with open(dataset_path, 'rb') as f:
    ds = pickle.load(f)

labels = ds['labels']
tx_grids = ds['tx_grids']

print(f"✓ Loaded {len(labels)} samples")

# Get first benign and first attack
benign_idx = np.where(labels == 0)[0][0]
attack_idx = np.where(labels == 1)[0][0]

print(f"\n📊 Analyzing samples:")
print(f"  Benign index: {benign_idx}")
print(f"  Attack index: {attack_idx}")

# Extract grids (squeeze to [symbols, subcarriers])
b_grid = np.squeeze(tx_grids[benign_idx])  # Should be [10, 64]
a_grid = np.squeeze(tx_grids[attack_idx])

print(f"\n🔍 DEBUG tx_grids shape = {np.array(tx_grids).shape}")
print(f"  Benign grid shape: {b_grid.shape}")
print(f"  Attack grid shape: {a_grid.shape}")

# Check if shapes match expected [symbols, subcarriers]
if b_grid.ndim != 2 or a_grid.ndim != 2:
    print(f"⚠️ WARNING: Grid shapes unexpected! Expected 2D [symbols, subs], got {b_grid.shape}")
    # Try to fix
    if b_grid.ndim > 2:
        b_grid = b_grid.reshape(-1, b_grid.shape[-1])
        a_grid = a_grid.reshape(-1, a_grid.shape[-1])
        print(f"  Reshaped to: {b_grid.shape}")

# ===== مورد 3: تفاوت طیفی =====
print(f"\n{'='*70}")
print("🔍 مورد 3: تفاوت طیفی (Spectral Difference)")
print(f"{'='*70}")

# Magnitude difference
mag_b = np.abs(b_grid)
mag_a = np.abs(a_grid)
delta_mag = mag_a - mag_b

print(f"  Δmag stats:")
print(f"    Mean:   {delta_mag.mean():.6f}")
print(f"    Std:    {delta_mag.std():.6f}")
print(f"    Max:    {delta_mag.max():.6f}")
print(f"    Min:    {delta_mag.min():.6f}")

# Check if difference is visible
if np.abs(delta_mag.mean()) < 0.01:
    print(f"  ⚠️ WARNING: تفاوت خیلی کم است! باید بیشتر باشد.")
else:
    print(f"  ✅ تفاوت قابل مشاهده است")

# Power difference
power_b = np.mean(np.abs(b_grid)**2)
power_a = np.mean(np.abs(a_grid)**2)
power_diff = (power_a - power_b) / power_b * 100

print(f"\n  Power analysis:")
print(f"    Benign power: {power_b:.6e}")
print(f"    Attack power: {power_a:.6e}")
print(f"    Difference:   {power_diff:.2f}%")

# ===== مورد 9: Energy Ratio با Focus Mask =====
print(f"\n{'='*70}")
print("🔍 مورد 9: Energy Ratio (Mask vs Non-Mask)")
print(f"{'='*70}")

# Build focus mask (same logic as detector)
n_sym, n_sc = b_grid.shape

# Expected injection: symbols [1,2,3,4,5,6,7], first half subcarriers
if n_sym >= 10:
    mask_symbols = list(range(1, min(n_sym-1, 8)))
elif n_sym >= 7:
    mask_symbols = [1,2,3,4]
else:
    mask_symbols = [1,2]

mask_subs = np.arange(n_sc // 2)  # First half

print(f"  Building focus mask:")
print(f"    Expected injection symbols: {mask_symbols}")
print(f"    Expected injection subcarriers: [0..{n_sc//2-1}]")

# Create mask
focus_mask = np.zeros((n_sym, n_sc), dtype=bool)
for s in mask_symbols:
    if s < n_sym:
        focus_mask[s, mask_subs] = True

print(f"    Mask nonzero count: {np.count_nonzero(focus_mask)}")

# Compute energy ratio
M = focus_mask
delta_abs = np.abs(delta_mag)

energy_in = np.mean(delta_abs[M])
energy_out = np.mean(delta_abs[~M])
ratio = energy_in / (energy_out + 1e-9)

print(f"\n  Energy distribution:")
print(f"    Inside mask (injection region):  {energy_in:.6f}")
print(f"    Outside mask:                    {energy_out:.6f}")
print(f"    Ratio (in/out):                  {ratio:.3f}")

if ratio > 1.3:
    print(f"    ✅ PASS: Ratio > 1.3 → Mask aligned with injection!")
elif ratio > 1.0:
    print(f"    ⚠️ WEAK: Ratio slightly > 1.0 → Partial alignment")
else:
    print(f"    ❌ FAIL: Ratio ≤ 1.0 → Mask NOT aligned with injection!")
    print(f"           → باید mask را دقیق با اندیس‌های واقعی injection بسازی")

# Additional: Show where maximum difference is
max_idx = np.unravel_index(np.argmax(delta_abs), delta_abs.shape)
print(f"\n  Maximum difference at: symbol={max_idx[0]}, subcarrier={max_idx[1]}")
print(f"    Is in mask? {focus_mask[max_idx]}")

# ===== Summary =====
print(f"\n{'='*70}")
print("📋 SUMMARY")
print(f"{'='*70}")
print(f"  ✓ Power difference: {power_diff:.2f}%")
print(f"  ✓ Mean magnitude Δ: {delta_mag.mean():.6f}")
print(f"  ✓ Energy ratio: {ratio:.3f}")

if power_diff > 3.0 and ratio > 1.3:
    print(f"\n  ✅ Everything looks good!")
elif power_diff < 3.0:
    print(f"\n  ⚠️ Power difference too low → Increase COVERT_AMP")
elif ratio < 1.3:
    print(f"\n  ⚠️ Mask misalignment → Fix focus_mask in detector")
else:
    print(f"\n  ⚠️ Check other issues (noise, feature extraction, etc.)")

print(f"{'='*70}\n")
