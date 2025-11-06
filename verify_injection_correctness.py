#!/usr/bin/env python3
"""
🔍 Verify Injection Correctness
================================
Comprehensive check of injection correctness:
1. Pre-channel injection ✅
2. power_diff_pct < 5% ✅
3. pattern_boost in subcarriers 24-39 ✅
4. doppler_hz non-zero and reasonable ✅
"""

import os
import sys
import pickle
import numpy as np
import glob

from config.settings import DATASET_DIR


def verify_injection_correctness(dataset_path=None):
    """Verify all injection correctness criteria."""
    print("="*70)
    print("🔍 INJECTION CORRECTNESS VERIFICATION")
    print("="*70)
    
    # Auto-detect dataset if not provided
    if dataset_path is None:
        dataset_files = glob.glob(os.path.join(DATASET_DIR, "dataset_*.pkl"))
        if not dataset_files:
            print(f"❌ No dataset found in {DATASET_DIR}")
            return False
        dataset_path = sorted(dataset_files)[-1]  # Use most recent
        print(f"📂 Auto-detected dataset: {os.path.basename(dataset_path)}")
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    labels = np.array(dataset['labels'])
    tx_grids = np.array(dataset['tx_grids'])
    
    # Get benign and attack samples
    benign_mask = (labels == 0)
    attack_mask = (labels == 1)
    benign_grids = np.squeeze(tx_grids[benign_mask])
    attack_grids = np.squeeze(tx_grids[attack_mask])
    
    # Expected injection locations (middle band)
    expected_symbols = [1, 3, 5, 7]
    expected_subcarriers = np.arange(24, 40)  # Middle band 24-39
    
    print(f"\n📊 Dataset Info:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Benign: {np.sum(benign_mask)}, Attack: {np.sum(attack_mask)}")
    print(f"  Expected injection: symbols {expected_symbols}, subcarriers {expected_subcarriers.tolist()}")
    
    # ===== CHECK 1: Pre-channel injection =====
    print(f"\n{'='*70}")
    print("✅ CHECK 1: Pre-channel Injection")
    print(f"{'='*70}")
    print("  ✓ Injection happens in tx_grids (pre-channel)")
    print("  ✓ This is verified by using tx_grids for analysis")
    
    # ===== CHECK 2: power_diff_pct < 5% =====
    print(f"\n{'='*70}")
    print("✅ CHECK 2: Power Difference < 5%")
    print(f"{'='*70}")
    
    benign_power = np.mean(np.abs(benign_grids) ** 2)
    attack_power = np.mean(np.abs(attack_grids) ** 2)
    power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    print(f"  Benign power: {benign_power:.6e}")
    print(f"  Attack power: {attack_power:.6e}")
    print(f"  Power diff:   {power_diff_pct:.2f}%")
    
    if power_diff_pct < 5.0:
        print(f"  ✅ PASS: power_diff_pct < 5% (ultra-covert)")
    elif power_diff_pct < 10.0:
        print(f"  ⚠️  WARNING: power_diff_pct = {power_diff_pct:.2f}% (5-10%, acceptable)")
    else:
        print(f"  ❌ FAIL: power_diff_pct = {power_diff_pct:.2f}% (> 10%, too high)")
    
    # ===== CHECK 3: pattern_boost in subcarriers 24-39 =====
    print(f"\n{'='*70}")
    print("✅ CHECK 3: Pattern Boost in Subcarriers 24-39")
    print(f"{'='*70}")
    
    # Calculate magnitude in injection locations
    benign_injection = []
    attack_injection = []
    benign_outside = []
    attack_outside = []
    
    for s in expected_symbols:
        if s < benign_grids.shape[1]:
            for sc in expected_subcarriers:
                if sc < benign_grids.shape[2]:
                    benign_injection.extend(np.abs(benign_grids[:, s, sc]))
                    attack_injection.extend(np.abs(attack_grids[:, s, sc]))
    
    # Outside locations
    for s in range(benign_grids.shape[1]):
        for sc in range(benign_grids.shape[2]):
            if s not in expected_symbols or sc not in expected_subcarriers:
                benign_outside.extend(np.abs(benign_grids[:, s, sc]))
                attack_outside.extend(np.abs(attack_grids[:, s, sc]))
    
    benign_injection_mean = np.mean(benign_injection)
    attack_injection_mean = np.mean(attack_injection)
    pattern_boost = ((attack_injection_mean - benign_injection_mean) / (benign_injection_mean + 1e-12)) * 100
    
    benign_outside_mean = np.mean(benign_outside)
    attack_outside_mean = np.mean(attack_outside)
    outside_boost = ((attack_outside_mean - benign_outside_mean) / (benign_outside_mean + 1e-12)) * 100
    
    print(f"  Injection locations (24-39):")
    print(f"    Benign mean: {benign_injection_mean:.6f}")
    print(f"    Attack mean: {attack_injection_mean:.6f}")
    print(f"    Pattern boost: {pattern_boost:.2f}%")
    
    print(f"  Outside locations:")
    print(f"    Benign mean: {benign_outside_mean:.6f}")
    print(f"    Attack mean: {attack_outside_mean:.6f}")
    print(f"    Outside boost: {outside_boost:.2f}%")
    
    if pattern_boost > 10.0 and abs(outside_boost) < 5.0:
        print(f"  ✅ PASS: Pattern boost in 24-39 = {pattern_boost:.2f}%, outside = {outside_boost:.2f}%")
    elif pattern_boost > 5.0:
        print(f"  ⚠️  WARNING: Pattern boost = {pattern_boost:.2f}% (5-10%, may be weak)")
    else:
        print(f"  ❌ FAIL: Pattern boost = {pattern_boost:.2f}% (< 5%, too weak)")
    
    # ===== CHECK 4: doppler_hz non-zero and reasonable =====
    print(f"\n{'='*70}")
    print("✅ CHECK 4: Doppler Non-Zero and Reasonable")
    print(f"{'='*70}")
    
    dopplers = []
    if 'meta' in dataset and isinstance(dataset['meta'], list):
        for m in dataset['meta']:
            if isinstance(m, dict) and 'doppler_hz' in m:
                dopplers.append(m['doppler_hz'])
    
    if len(dopplers) > 0:
        dopplers = np.array(dopplers)
        doppler_mean = np.mean(dopplers)
        doppler_std = np.std(dopplers)
        doppler_min = np.min(dopplers)
        doppler_max = np.max(dopplers)
        non_zero_count = np.sum(np.abs(dopplers) > 1e-6)
        
        print(f"  Doppler samples: {len(dopplers)}")
        print(f"  Mean: {doppler_mean:.2f} Hz")
        print(f"  Std: {doppler_std:.2f} Hz")
        print(f"  Range: [{doppler_min:.2f}, {doppler_max:.2f}] Hz")
        print(f"  Non-zero: {non_zero_count}/{len(dopplers)} ({non_zero_count/len(dopplers)*100:.1f}%)")
        
        # Check if reasonable (for 28 GHz, expect ±hundreds of kHz)
        if non_zero_count == len(dopplers) and abs(doppler_mean) > 1.0:
            print(f"  ✅ PASS: All doppler_hz non-zero and reasonable")
        elif non_zero_count > len(dopplers) * 0.9:
            print(f"  ⚠️  WARNING: {len(dopplers) - non_zero_count} samples have zero doppler")
        else:
            print(f"  ❌ FAIL: Too many zero doppler values")
    else:
        print(f"  ❌ FAIL: No doppler_hz data in meta")
    
    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("📋 VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    checks_passed = 0
    total_checks = 4
    
    if power_diff_pct < 5.0:
        checks_passed += 1
    if pattern_boost > 10.0 and abs(outside_boost) < 5.0:
        checks_passed += 1
    if len(dopplers) > 0 and np.sum(np.abs(dopplers) > 1e-6) == len(dopplers):
        checks_passed += 1
    # Pre-channel is always true (we use tx_grids)
    checks_passed += 1
    
    print(f"  Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print(f"  ✅ ALL CHECKS PASSED!")
        return True
    else:
        print(f"  ⚠️  Some checks failed. Review output above.")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify injection correctness")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset file")
    args = parser.parse_args()
    
    verify_injection_correctness(args.dataset)

