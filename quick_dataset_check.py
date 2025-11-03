#!/usr/bin/env python3
"""
Quick Dataset Sanity Check
===========================
Fast validation of critical dataset properties.
"""

import os
import pickle
import numpy as np
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

def quick_check():
    """Quick sanity check of dataset."""
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    print("🔍 Quick Dataset Check")
    print("-" * 40)
    
    # Load
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    labels = dataset['labels']
    iq_samples = dataset['iq_samples']
    
    # Basic checks
    n_total = len(labels)
    n_benign = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    
    print(f"✓ Samples: {n_total} ({n_benign} benign, {n_attack} attack)")
    
    # NaN check
    has_nan = any(np.any(np.isnan(s)) for s in iq_samples)
    has_inf = any(np.any(np.isinf(s)) for s in iq_samples)
    
    if has_nan or has_inf:
        print(f"❌ Found NaN/Inf in signals!")
        return False
    print("✓ No NaN/Inf values")
    
    # Power analysis (detailed)
    benign_powers = [np.mean(np.abs(iq_samples[i])**2) for i in range(n_total) if labels[i] == 0]
    attack_powers = [np.mean(np.abs(iq_samples[i])**2) for i in range(n_total) if labels[i] == 1]
    
    benign_mean = np.mean(benign_powers)
    benign_std = np.std(benign_powers)
    attack_mean = np.mean(attack_powers)
    attack_std = np.std(attack_powers)
    power_ratio = attack_mean / benign_mean
    
    # Coefficient of variation
    cv_benign = benign_std / benign_mean if benign_mean > 0 else 0
    cv_attack = attack_std / attack_mean if attack_mean > 0 else 0
    
    print(f"✓ Power ratio (attack/benign): {power_ratio:.4f}")
    print(f"  Benign: {benign_mean:.6e} ± {benign_std:.6e} (CV={cv_benign:.2f})")
    print(f"  Attack: {attack_mean:.6e} ± {attack_std:.6e} (CV={cv_attack:.2f})")
    
    # Effect size
    pooled_std = np.sqrt((np.var(benign_powers) + np.var(attack_powers)) / 2)
    cohen_d = abs(benign_mean - attack_mean) / pooled_std if pooled_std > 0 else 0
    
    print(f"✓ Effect size (Cohen's d): {cohen_d:.4f}")
    
    # Power normalization check
    if 0.95 <= power_ratio <= 1.05:
        print("  ✅ Excellent: Truly covert (power ≈ 1.0)")
    elif 0.90 <= power_ratio <= 1.10:
        print("  ✅ Good: Mostly covert (power close to 1.0)")
    elif 0.80 <= power_ratio <= 1.20:
        print("  ⚠️  Warning: Somewhat detectable by power")
    else:
        print("  ❌ Poor: Easily detectable by power!")
    
    # Satellite altitudes
    # NOTE: 'position' in dataset is local coordinates [x_local, y_local, altitude]
    # The z-coordinate IS the altitude (not ECEF!)
    altitudes = []
    for sample_sats in dataset['satellite_receptions']:
        if sample_sats:
            for sat in sample_sats:
                if 'position' in sat:
                    pos = sat['position']
                    altitude_km = pos[2] / 1e3  # z-coordinate IS altitude in local coords
                    altitudes.append(altitude_km)
    
    if altitudes:
        alt_min = np.min(altitudes)
        alt_max = np.max(altitudes)
        print(f"✓ Altitudes: {alt_min:.1f} - {alt_max:.1f} km")
        
        if alt_min < 0:
            print(f"❌ Negative altitudes detected!")
            return False
    
    # Verdict
    print("-" * 40)
    
    if cohen_d < 0.01:
        print("❌ FAIL: Classes are identical!")
        print("   → Covert signal not injected?")
        return False
    elif cohen_d < 0.1:
        print("⚠️ WARNING: Very weak signal")
        print("   → Detection will be very difficult")
    else:
        print(f"✅ PASS: Dataset looks good (d={cohen_d:.4f})")
    
    return True


if __name__ == "__main__":
    import sys
    success = quick_check()
    sys.exit(0 if success else 1)
