#!/usr/bin/env python3
"""
Test Dataset Quality
====================
Comprehensive tests to validate dataset integrity and quality.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

def test_dataset_quality():
    """Run comprehensive dataset quality tests."""
    
    print("="*60)
    print("DATASET QUALITY TEST")
    print("="*60)
    
    # Load dataset
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("Please generate dataset first:")
        print("  python3 generate_dataset_parallel.py")
        return False
    
    print(f"\n[1/10] Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print("✓ Dataset loaded successfully")
    
    # Test 1: Basic structure
    print("\n[2/10] Testing dataset structure...")
    required_keys = ['iq_samples', 'labels', 'emitter_locations', 'satellite_receptions']
    missing_keys = [k for k in required_keys if k not in dataset]
    if missing_keys:
        print(f"❌ FAIL: Missing keys: {missing_keys}")
        return False
    print(f"✓ All required keys present: {list(dataset.keys())}")
    
    # Test 2: Sample counts
    print("\n[3/10] Testing sample counts...")
    labels = dataset['labels']
    n_benign = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    n_total = len(labels)
    
    print(f"  Total samples: {n_total}")
    print(f"  Benign (0): {n_benign}")
    print(f"  Attack (1): {n_attack}")
    
    if n_benign == 0 or n_attack == 0:
        print("❌ FAIL: One class has zero samples!")
        return False
    
    balance_ratio = min(n_benign, n_attack) / max(n_benign, n_attack)
    if balance_ratio < 0.9:
        print(f"⚠️ WARNING: Class imbalance detected (ratio={balance_ratio:.2f})")
    else:
        print(f"✓ Classes balanced (ratio={balance_ratio:.2f})")
    
    # Test 3: NaN/Inf check
    print("\n[4/10] Testing for NaN/Inf values...")
    nan_count = 0
    inf_count = 0
    
    iq_samples = dataset['iq_samples']
    for i, sample in enumerate(iq_samples):
        if np.any(np.isnan(sample)):
            nan_count += 1
        if np.any(np.isinf(sample)):
            inf_count += 1
    
    if nan_count > 0:
        print(f"❌ FAIL: {nan_count}/{len(iq_samples)} samples contain NaN!")
        return False
    if inf_count > 0:
        print(f"❌ FAIL: {inf_count}/{len(iq_samples)} samples contain Inf!")
        return False
    print("✓ No NaN/Inf values found")
    
    # Test 4: Signal power analysis
    print("\n[5/10] Analyzing signal power...")
    benign_powers = []
    attack_powers = []
    
    for i, label in enumerate(labels):
        power = np.mean(np.abs(iq_samples[i])**2)
        if label == 0:
            benign_powers.append(power)
        else:
            attack_powers.append(power)
    
    benign_mean = np.mean(benign_powers)
    attack_mean = np.mean(attack_powers)
    power_ratio = attack_mean / benign_mean if benign_mean > 0 else 0
    
    print(f"  Benign power: {benign_mean:.6e}")
    print(f"  Attack power: {attack_mean:.6e}")
    print(f"  Power ratio (attack/benign): {power_ratio:.4f}")
    
    if power_ratio < 0.9 or power_ratio > 1.1:
        print(f"⚠️ WARNING: Significant power difference detected!")
        print(f"  This might affect detection. Expected ratio ≈ 1.0")
    else:
        print("✓ Power levels similar between classes")
    
    # Test 5: SNR analysis
    print("\n[6/10] Analyzing SNR distribution...")
    if 'satellite_receptions' in dataset:
        snrs = []
        for sample_sats in dataset['satellite_receptions']:
            if sample_sats:
                for sat in sample_sats:
                    if 'ebno_dB' in sat:
                        snrs.append(sat['ebno_dB'])
        
        if snrs:
            snr_mean = np.mean(snrs)
            snr_std = np.std(snrs)
            snr_min = np.min(snrs)
            snr_max = np.max(snrs)
            
            print(f"  SNR mean: {snr_mean:.2f} dB")
            print(f"  SNR std: {snr_std:.2f} dB")
            print(f"  SNR range: [{snr_min:.2f}, {snr_max:.2f}] dB")
            
            if snr_mean < 5:
                print("⚠️ WARNING: Low SNR detected (mean < 5 dB)")
            elif snr_mean > 30:
                print("⚠️ WARNING: Very high SNR (mean > 30 dB) - unrealistic?")
            else:
                print("✓ SNR levels reasonable")
        else:
            print("⚠️ WARNING: No SNR information found")
    
    # Test 6: Emitter locations
    print("\n[7/10] Testing emitter locations...")
    emitter_locs = dataset['emitter_locations']
    valid_locs = [loc for loc in emitter_locs if loc is not None]
    
    print(f"  Total emitter locations: {len(valid_locs)}/{len(emitter_locs)}")
    
    if len(valid_locs) == 0:
        print("❌ FAIL: No valid emitter locations!")
        return False
    
    # Check if locations are reasonable (within ±800km)
    max_coord = 800e3  # 800 km
    invalid_locs = 0
    for loc in valid_locs:
        if np.any(np.abs(loc) > max_coord):
            invalid_locs += 1
    
    if invalid_locs > 0:
        print(f"⚠️ WARNING: {invalid_locs} locations outside ±800km range")
    else:
        print("✓ All emitter locations within reasonable bounds")
    
    # Test 7: Satellite positions
    print("\n[8/10] Testing satellite positions...")
    sat_altitudes = []
    
    # NOTE: 'position' in dataset is local coordinates [x_local, y_local, altitude]
    # The z-coordinate IS the altitude (not ECEF!)
    for sample_sats in dataset['satellite_receptions']:
        if sample_sats:
            for sat in sample_sats:
                if 'position' in sat:
                    pos = sat['position']
                    altitude = pos[2]  # z-coordinate IS altitude in local coords
                    sat_altitudes.append(altitude)
    
    if sat_altitudes:
        alt_mean = np.mean(sat_altitudes) / 1e3  # km
        alt_min = np.min(sat_altitudes) / 1e3
        alt_max = np.max(sat_altitudes) / 1e3
        
        print(f"  Altitude mean: {alt_mean:.1f} km")
        print(f"  Altitude range: [{alt_min:.1f}, {alt_max:.1f}] km")
        
        # Check for negative altitudes
        negative_alts = [a for a in sat_altitudes if a < 0]
        if negative_alts:
            print(f"❌ FAIL: {len(negative_alts)} satellites have negative altitude!")
            return False
        
        # Check for unrealistic altitudes
        if alt_min < 200 or alt_max > 2000:
            print("⚠️ WARNING: Altitudes outside typical LEO range (200-2000 km)")
        else:
            print("✓ Satellite altitudes in LEO range")
    else:
        print("⚠️ WARNING: No satellite position information")
    
    # Test 8: Signal length consistency
    print("\n[9/10] Testing signal length consistency...")
    signal_lengths = [len(s) for s in iq_samples]
    unique_lengths = np.unique(signal_lengths)
    
    if len(unique_lengths) > 1:
        print(f"  Found {len(unique_lengths)} different signal lengths:")
        print(f"  Min: {np.min(signal_lengths)}, Max: {np.max(signal_lengths)}")
        print("  ℹ️ Note: Variable lengths are OK if properly handled")
    else:
        print(f"✓ All signals have same length: {signal_lengths[0]}")
    
    # Test 9: Feature discriminability
    print("\n[10/10] Testing feature discriminability...")
    
    # Simple power-based separability test
    benign_powers_arr = np.array(benign_powers)
    attack_powers_arr = np.array(attack_powers)
    
    # T-test equivalent: mean difference normalized by pooled std
    pooled_std = np.sqrt((np.var(benign_powers_arr) + np.var(attack_powers_arr)) / 2)
    cohen_d = abs(benign_mean - attack_mean) / pooled_std if pooled_std > 0 else 0
    
    print(f"  Cohen's d (effect size): {cohen_d:.4f}")
    
    if cohen_d < 0.01:
        print("❌ FAIL: Classes appear identical (d < 0.01)!")
        print("  → Covert signal may not be injected properly")
        return False
    elif cohen_d < 0.1:
        print("⚠️ WARNING: Very small effect size (d < 0.1)")
        print("  → Detection will be very difficult")
    elif cohen_d < 0.5:
        print("⚠️ WARNING: Small effect size (d < 0.5)")
        print("  → Detection challenging but possible")
    else:
        print(f"✓ Detectable difference between classes (d = {cohen_d:.4f})")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET QUALITY SUMMARY")
    print("="*60)
    print(f"✓ Total samples: {n_total}")
    print(f"✓ No NaN/Inf values")
    print(f"✓ Satellite altitudes: {alt_min:.1f} - {alt_max:.1f} km")
    print(f"✓ SNR mean: {snr_mean:.2f} dB" if snrs else "⚠️ No SNR data")
    print(f"{'✓' if cohen_d > 0.1 else '⚠️'} Effect size: {cohen_d:.4f}")
    print(f"{'✓' if 0.9 < power_ratio < 1.1 else '⚠️'} Power ratio: {power_ratio:.4f}")
    
    # Overall verdict
    print("\n" + "="*60)
    if cohen_d < 0.01:
        print("❌ OVERALL: FAIL - Dataset appears corrupted")
        print("   Recommendation: Re-generate dataset")
        return False
    elif cohen_d < 0.1 or not (0.8 < power_ratio < 1.2):
        print("⚠️ OVERALL: WARNING - Dataset has quality issues")
        print("   Recommendation: Check covert injection parameters")
        return True  # Pass but with warnings
    else:
        print("✅ OVERALL: PASS - Dataset quality is good")
        return True


def plot_distributions(dataset):
    """Plot power distributions for visual inspection."""
    print("\n[Bonus] Generating distribution plots...")
    
    labels = dataset['labels']
    iq_samples = dataset['iq_samples']
    
    benign_powers = []
    attack_powers = []
    
    for i, label in enumerate(labels):
        power = np.mean(np.abs(iq_samples[i])**2)
        if label == 0:
            benign_powers.append(power)
        else:
            attack_powers.append(power)
    
    plt.figure(figsize=(12, 5))
    
    # Power histogram
    plt.subplot(1, 2, 1)
    plt.hist(benign_powers, bins=50, alpha=0.6, label='Benign', density=True)
    plt.hist(attack_powers, bins=50, alpha=0.6, label='Attack', density=True)
    plt.xlabel('Signal Power')
    plt.ylabel('Density')
    plt.title('Power Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([benign_powers, attack_powers], labels=['Benign', 'Attack'])
    plt.ylabel('Signal Power')
    plt.title('Power Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'result/dataset_quality_check.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    try:
        success = test_dataset_quality()
        
        # Load dataset for plotting
        dataset_path = (
            f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
            f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
        )
        
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            plot_distributions(dataset)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ ERROR: Test script failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
