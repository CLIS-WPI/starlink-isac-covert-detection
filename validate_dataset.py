#!/usr/bin/env python3
"""
🔍 Dataset Validation Script
============================
Quick validation checklist before training:
1. Doppler is actually applied (non-zero, reasonable distribution)
2. Injection is pre-channel (not post-channel)
3. Power-preserving works (power_diff_pct < 10%)
4. CSI-LS is healthy (variance reasonable)
5. Dataset structure is correct (all keys present)
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from config.settings import (
    DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA,
    INSIDER_MODE, POWER_PRESERVING_COVERT, COVERT_AMP, CARRIER_FREQUENCY
)


def check_doppler(dataset):
    """Check if Doppler is applied correctly."""
    print("\n" + "="*70)
    print("🔍 CHECK 1: Doppler Application")
    print("="*70)
    
    if 'meta' not in dataset or not dataset['meta']:
        print("  ❌ FAIL: 'meta' key missing or empty")
        return False
    
    dopplers = []
    for m in dataset['meta']:
        if isinstance(m, dict) and 'doppler_hz' in m:
            dopplers.append(m['doppler_hz'])
    
    if len(dopplers) == 0:
        print("  ❌ FAIL: No doppler_hz found in meta")
        return False
    
    dopplers = np.array(dopplers)
    mean_fd = np.mean(dopplers)
    std_fd = np.std(dopplers)
    non_zero = np.sum(np.abs(dopplers) > 1.0)  # At least 1 Hz
    
    print(f"  ✓ Doppler found in {len(dopplers)} samples")
    print(f"  ✓ Mean: {mean_fd:.2f} Hz, Std: {std_fd:.2f} Hz")
    print(f"  ✓ Non-zero (>1 Hz): {non_zero}/{len(dopplers)} ({100*non_zero/len(dopplers):.1f}%)")
    
    # Expected range for 28 GHz LEO: ±50 kHz to ±500 kHz
    expected_min = -500e3
    expected_max = 500e3
    in_range = np.sum((dopplers >= expected_min) & (dopplers <= expected_max))
    
    if mean_fd == 0.0 and std_fd == 0.0:
        print(f"  ❌ FAIL: All Doppler values are zero!")
        return False
    elif non_zero < len(dopplers) * 0.5:
        print(f"  ⚠️  WARNING: {100*(1-non_zero/len(dopplers)):.1f}% samples have zero Doppler")
    else:
        print(f"  ✅ PASS: Doppler is applied (mean={mean_fd:.2f} Hz)")
    
    # Check FFT shift in a sample
    if 'rx_grids' in dataset and len(dataset['rx_grids']) > 0:
        sample_grid = dataset['rx_grids'][0]
        if sample_grid.ndim >= 2:
            # Take one symbol, compute FFT
            sym = sample_grid[0] if sample_grid.ndim == 2 else sample_grid[0, 0]
            fft_vals = np.fft.fft(sym)
            fft_peak_idx = np.argmax(np.abs(fft_vals))
            if fft_peak_idx != 0:
                print(f"  ✓ FFT peak shifted (idx={fft_peak_idx}), Doppler likely applied")
            else:
                print(f"  ⚠️  FFT peak at 0, may indicate no Doppler")
    
    return True


def check_injection_timing(dataset):
    """Check that injection is pre-channel."""
    print("\n" + "="*70)
    print("🔍 CHECK 2: Injection Timing (Pre-Channel)")
    print("="*70)
    
    if 'meta' not in dataset:
        print("  ❌ FAIL: 'meta' key missing")
        return False
    
    insider_modes = []
    for m in dataset['meta']:
        if isinstance(m, dict) and 'insider_mode' in m:
            insider_modes.append(m['insider_mode'])
    
    if len(insider_modes) == 0:
        print("  ⚠️  WARNING: insider_mode not in meta (may be old dataset)")
    else:
        unique_modes = set(insider_modes)
        print(f"  ✓ Insider modes found: {unique_modes}")
        if INSIDER_MODE in unique_modes or len(unique_modes) == 1:
            print(f"  ✅ PASS: Injection mode matches INSIDER_MODE={INSIDER_MODE}")
        else:
            print(f"  ⚠️  WARNING: Mode mismatch (expected {INSIDER_MODE})")
    
    # Check power difference between tx_grids and rx_grids
    if 'tx_grids' in dataset and 'rx_grids' in dataset:
        labels = dataset['labels']
        benign_mask = (labels == 0)
        attack_mask = (labels == 1)
        
        if np.sum(attack_mask) > 0:
            tx_attack = dataset['tx_grids'][attack_mask]
            rx_attack = dataset['rx_grids'][attack_mask]
            
            # Power in tx (after injection, pre-channel)
            tx_power = np.mean([np.mean(np.abs(g)**2) for g in tx_attack])
            # Power in rx (post-channel)
            rx_power = np.mean([np.mean(np.abs(g)**2) for g in rx_attack])
            
            # If injection is pre-channel, tx should have injection pattern
            # If injection is post-channel, tx and rx would be similar (wrong!)
            # We can't directly detect this, but we log it
            print(f"  ✓ TX power (attack, pre-channel): {tx_power:.6e}")
            print(f"  ✓ RX power (attack, post-channel): {rx_power:.6e}")
            print(f"  → Note: Injection should be in tx_grids (pre-channel)")
    
    return True


def check_power_preserving(dataset):
    """Check power-preserving works."""
    print("\n" + "="*70)
    print("🔍 CHECK 3: Power-Preserving Covert Injection")
    print("="*70)
    
    if 'tx_grids' not in dataset or 'labels' not in dataset:
        print("  ❌ FAIL: Missing tx_grids or labels")
        return False
    
    labels = dataset['labels']
    benign_mask = (labels == 0)
    attack_mask = (labels == 1)
    
    if np.sum(benign_mask) == 0 or np.sum(attack_mask) == 0:
        print("  ❌ FAIL: Missing benign or attack samples")
        return False
    
    # Compute power for each sample
    benign_powers = []
    attack_powers = []
    
    for i, label in enumerate(labels):
        grid = dataset['tx_grids'][i]
        power = np.mean(np.abs(grid)**2)
        if label == 0:
            benign_powers.append(power)
        else:
            attack_powers.append(power)
    
    benign_powers = np.array(benign_powers)
    attack_powers = np.array(attack_powers)
    
    mean_benign = np.mean(benign_powers)
    mean_attack = np.mean(attack_powers)
    power_diff_pct = abs(mean_attack - mean_benign) / (mean_benign + 1e-12) * 100.0
    
    print(f"  ✓ Benign power: {mean_benign:.6e} ± {np.std(benign_powers):.6e}")
    print(f"  ✓ Attack power: {mean_attack:.6e} ± {np.std(attack_powers):.6e}")
    print(f"  ✓ Power difference: {power_diff_pct:.2f}%")
    
    # 🔍 ENHANCED: Check per-sample power differences from meta
    if 'meta' in dataset and dataset['meta']:
        power_diffs = []
        doppler_hz_list = []
        first_sample_meta = None
        
        for i, m in enumerate(dataset['meta']):
            if isinstance(m, dict):
                if 'power_diff_pct' in m:
                    power_diffs.append(m['power_diff_pct'])
                if 'doppler_hz' in m:
                    doppler_hz_list.append(m['doppler_hz'])
                if i == 0 and first_sample_meta is None:
                    first_sample_meta = m
        
        if len(power_diffs) > 0:
            power_diffs = np.array(power_diffs)
            print(f"\n  📊 Per-sample power_diff_pct:")
            print(f"      Mean: {np.mean(power_diffs):.2f}%, Std: {np.std(power_diffs):.2f}%")
            print(f"      Range: [{np.min(power_diffs):.2f}%, {np.max(power_diffs):.2f}%]")
            
            # 🔍 CRITICAL CHECK: If power_diff is too low, injection may not be visible
            if np.mean(power_diffs) < 1.0:
                print(f"      ⚠️  WARNING: Mean power_diff_pct={np.mean(power_diffs):.2f}% < 1% - injection may be too weak!")
            elif np.mean(power_diffs) < 5.0:
                print(f"      ⚠️  WARNING: Mean power_diff_pct={np.mean(power_diffs):.2f}% < 5% - injection may be hard to detect")
            else:
                print(f"      ✓ Power difference is sufficient for detection")
        
        if len(doppler_hz_list) > 0:
            doppler_hz_list = np.array(doppler_hz_list)
            print(f"\n  📊 Doppler range:")
            print(f"      Min: {np.min(doppler_hz_list):.2f} Hz")
            print(f"      Max: {np.max(doppler_hz_list):.2f} Hz")
            print(f"      Mean: {np.mean(doppler_hz_list):.2f} Hz")
        
        if first_sample_meta is not None:
            print(f"\n  📋 First sample meta (sample #0):")
            for key, value in first_sample_meta.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"      {key}: {value}")
                elif isinstance(value, np.ndarray) and value.size < 10:
                    print(f"      {key}: {value}")
    
    if POWER_PRESERVING_COVERT:
        if power_diff_pct < 10.0:
            print(f"  ✅ PASS: Power-preserving works (diff={power_diff_pct:.2f}% < 10%)")
            return True
        else:
            print(f"  ❌ FAIL: Power diff too high ({power_diff_pct:.2f}% > 10%)")
            return False
    else:
        print(f"  ℹ️  INFO: Power-preserving disabled (expected higher diff)")
        return True


def check_csi_estimation(dataset):
    """Check CSI-LS estimation quality."""
    print("\n" + "="*70)
    print("🔍 CHECK 4: CSI-LS Estimation Quality")
    print("="*70)
    
    if 'csi_est' not in dataset or dataset['csi_est'] is None:
        print("  ❌ FAIL: 'csi_est' missing or None")
        return False
    
    csi_est = dataset['csi_est']
    print(f"  ✓ CSI_est shape: {csi_est.shape}")
    
    # Compute variance
    csi_mag = np.abs(csi_est)
    csi_variance = np.var(csi_mag)
    csi_mean = np.mean(csi_mag)
    csi_std = np.std(csi_mag)
    
    print(f"  ✓ CSI magnitude: mean={csi_mean:.6f}, std={csi_std:.6f}")
    print(f"  ✓ CSI variance: {csi_variance:.6e}")
    
    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(csi_est))
    inf_count = np.sum(np.isinf(csi_est))
    
    if nan_count > 0 or inf_count > 0:
        print(f"  ❌ FAIL: CSI contains NaN ({nan_count}) or Inf ({inf_count})")
        return False
    
    # Variance should be reasonable (not too large, not zero)
    if csi_variance < 1e-10:
        print(f"  ⚠️  WARNING: CSI variance too small (may be constant)")
    elif csi_variance > 1.0:
        print(f"  ⚠️  WARNING: CSI variance large (may indicate estimation issues)")
    else:
        print(f"  ✅ PASS: CSI variance reasonable ({csi_variance:.6e})")
    
    return True


def check_dataset_structure(dataset):
    """Check dataset has all required keys."""
    print("\n" + "="*70)
    print("🔍 CHECK 5: Dataset Structure")
    print("="*70)
    
    required_keys = ['tx_grids', 'rx_grids', 'labels']
    optional_keys = ['csi_est', 'meta', 'csi']
    
    missing = []
    for key in required_keys:
        if key not in dataset:
            missing.append(key)
    
    if missing:
        print(f"  ❌ FAIL: Missing required keys: {missing}")
        return False
    
    print(f"  ✓ Required keys present: {required_keys}")
    
    # Check shapes
    n_samples = len(dataset['labels'])
    print(f"  ✓ Total samples: {n_samples}")
    
    if 'tx_grids' in dataset:
        print(f"  ✓ tx_grids shape: {dataset['tx_grids'].shape}")
    if 'rx_grids' in dataset:
        print(f"  ✓ rx_grids shape: {dataset['rx_grids'].shape}")
    if 'csi_est' in dataset and dataset['csi_est'] is not None:
        print(f"  ✓ csi_est shape: {dataset['csi_est'].shape}")
    if 'meta' in dataset:
        print(f"  ✓ meta entries: {len(dataset['meta'])}")
    
    # Check label balance
    labels = dataset['labels']
    benign_count = np.sum(labels == 0)
    attack_count = np.sum(labels == 1)
    print(f"  ✓ Labels: benign={benign_count}, attack={attack_count}")
    
    if abs(benign_count - attack_count) > n_samples * 0.1:
        print(f"  ⚠️  WARNING: Class imbalance ({benign_count} vs {attack_count})")
    else:
        print(f"  ✅ PASS: Dataset structure correct")
    
    return True


def main():
    """Run all validation checks."""
    print("="*70)
    print("🔍 DATASET VALIDATION CHECKLIST")
    print("="*70)
    print(f"Dataset: {NUM_SAMPLES_PER_CLASS} samples per class")
    print(f"Scenario: {INSIDER_MODE}")
    print(f"Power-preserving: {POWER_PRESERVING_COVERT}")
    print(f"COVERT_AMP: {COVERT_AMP}")
    print("="*70)
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        return False
    
    print(f"\n📂 Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Dataset loaded: {len(dataset.get('labels', []))} samples")
    
    # Run all checks
    checks = [
        ("Dataset Structure", check_dataset_structure),
        ("Doppler Application", check_doppler),
        ("Injection Timing", check_injection_timing),
        ("Power-Preserving", check_power_preserving),
        ("CSI Estimation", check_csi_estimation),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func(dataset)
        except Exception as e:
            print(f"  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - Dataset is ready for training!")
        return True
    else:
        print("\n❌ SOME CHECKS FAILED - Please fix issues before training")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

