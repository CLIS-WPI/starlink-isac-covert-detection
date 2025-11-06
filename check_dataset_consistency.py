#!/usr/bin/env python3
"""
🔍 Dataset Consistency Checker
==============================
After merging datasets from multiple GPUs, check:
1. Power consistency (should be similar across GPUs)
2. Doppler consistency (should have reasonable distribution)
3. Labels consistency (should be balanced)
4. Meta consistency (power_diff_pct, doppler_hz, etc.)
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

from config.settings import (
    DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA,
    INSIDER_MODE, POWER_PRESERVING_COVERT, COVERT_AMP
)


def check_dataset_consistency(dataset_path):
    """Check consistency of merged dataset."""
    print("="*70)
    print("🔍 DATASET CONSISTENCY CHECKER")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Expected: {NUM_SAMPLES_PER_CLASS * 2} samples ({NUM_SAMPLES_PER_CLASS} per class)")
    print("="*70)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # ========================================================================
    # CHECK 1: Labels Consistency
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 CHECK 1: Labels Consistency")
    print("="*70)
    
    if 'labels' not in dataset:
        print("  ❌ FAIL: 'labels' key missing")
        return False
    
    labels = np.array(dataset['labels'])
    total_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"  ✓ Total samples: {total_samples}")
    print(f"  ✓ Unique labels: {unique_labels.tolist()}")
    print(f"  ✓ Label counts: {dict(zip(unique_labels, counts))}")
    
    # Check balance
    if len(unique_labels) != 2:
        print(f"  ❌ FAIL: Expected 2 classes, got {len(unique_labels)}")
        return False
    
    benign_count = counts[0] if unique_labels[0] == 0 else counts[1]
    attack_count = counts[1] if unique_labels[0] == 0 else counts[0]
    
    expected_benign = NUM_SAMPLES_PER_CLASS
    expected_attack = NUM_SAMPLES_PER_CLASS
    expected_total = expected_benign + expected_attack
    
    if total_samples != expected_total:
        print(f"  ⚠️  WARNING: Total samples mismatch!")
        print(f"      Expected: {expected_total}, Got: {total_samples}")
        print(f"      Difference: {abs(total_samples - expected_total)}")
    
    if benign_count != expected_benign or attack_count != expected_attack:
        print(f"  ⚠️  WARNING: Class imbalance!")
        print(f"      Expected: benign={expected_benign}, attack={expected_attack}")
        print(f"      Got: benign={benign_count}, attack={attack_count}")
        print(f"      Difference: benign={abs(benign_count - expected_benign)}, attack={abs(attack_count - expected_attack)}")
    else:
        print(f"  ✅ PASS: Labels balanced (benign={benign_count}, attack={attack_count})")
    
    # Check for label mixing (should not have alternating pattern)
    if total_samples > 100:
        first_100_labels = labels[:100]
        benign_ratio_first_100 = np.sum(first_100_labels == 0) / len(first_100_labels)
        if benign_ratio_first_100 < 0.3 or benign_ratio_first_100 > 0.7:
            print(f"  ⚠️  WARNING: First 100 samples may be from single GPU!")
            print(f"      Benign ratio in first 100: {benign_ratio_first_100:.2%}")
        else:
            print(f"  ✓ First 100 samples appear mixed (benign ratio: {benign_ratio_first_100:.2%})")
    
    # ========================================================================
    # CHECK 2: Power Consistency
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 CHECK 2: Power Consistency")
    print("="*70)
    
    if 'tx_grids' not in dataset:
        print("  ❌ FAIL: 'tx_grids' key missing")
        return False
    
    tx_grids = dataset['tx_grids']
    benign_mask = (labels == 0)
    attack_mask = (labels == 1)
    
    # Calculate power per sample
    benign_powers = []
    attack_powers = []
    
    for i in range(len(tx_grids)):
        grid = np.squeeze(tx_grids[i])
        power = np.mean(np.abs(grid)**2)
        if benign_mask[i]:
            benign_powers.append(power)
        else:
            attack_powers.append(power)
    
    benign_powers = np.array(benign_powers)
    attack_powers = np.array(attack_powers)
    
    print(f"  ✓ Benign power: mean={np.mean(benign_powers):.6e}, std={np.std(benign_powers):.6e}")
    print(f"  ✓ Attack power: mean={np.mean(attack_powers):.6e}, std={np.std(attack_powers):.6e}")
    
    # Check power difference
    power_diff_pct = abs(np.mean(attack_powers) - np.mean(benign_powers)) / (np.mean(benign_powers) + 1e-12) * 100.0
    print(f"  ✓ Power difference: {power_diff_pct:.2f}%")
    
    # Check power distribution consistency (should not have bimodal distribution)
    if len(benign_powers) > 50:
        benign_q25, benign_q75 = np.percentile(benign_powers, [25, 75])
        attack_q25, attack_q75 = np.percentile(attack_powers, [25, 75])
        
        print(f"  ✓ Benign power IQR: [{benign_q25:.6e}, {benign_q75:.6e}]")
        print(f"  ✓ Attack power IQR: [{attack_q25:.6e}, {attack_q75:.6e}]")
        
        # Check if distributions overlap significantly (good sign)
        overlap = min(benign_q75, attack_q75) - max(benign_q25, attack_q25)
        if overlap > 0:
            print(f"  ✓ Power distributions overlap (IQR overlap: {overlap:.6e})")
        else:
            print(f"  ⚠️  WARNING: Power distributions may be separated (IQR gap: {abs(overlap):.6e})")
    
    # Check for power outliers (may indicate GPU mismatch)
    benign_outliers = np.sum((benign_powers < np.percentile(benign_powers, 1)) | 
                            (benign_powers > np.percentile(benign_powers, 99)))
    attack_outliers = np.sum((attack_powers < np.percentile(attack_powers, 1)) | 
                             (attack_powers > np.percentile(attack_powers, 99)))
    
    if benign_outliers > len(benign_powers) * 0.05 or attack_outliers > len(attack_powers) * 0.05:
        print(f"  ⚠️  WARNING: High number of power outliers!")
        print(f"      Benign outliers: {benign_outliers}/{len(benign_powers)} ({100*benign_outliers/len(benign_powers):.1f}%)")
        print(f"      Attack outliers: {attack_outliers}/{len(attack_powers)} ({100*attack_outliers/len(attack_powers):.1f}%)")
    else:
        print(f"  ✓ Power outliers within normal range")
    
    # ========================================================================
    # CHECK 3: Doppler Consistency
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 CHECK 3: Doppler Consistency")
    print("="*70)
    
    if 'meta' not in dataset or not dataset['meta']:
        print("  ❌ FAIL: 'meta' key missing or empty")
        return False
    
    doppler_hz_list = []
    for m in dataset['meta']:
        if isinstance(m, dict) and 'doppler_hz' in m:
            doppler_hz_list.append(m['doppler_hz'])
    
    if len(doppler_hz_list) == 0:
        print("  ❌ FAIL: No doppler_hz found in meta")
        return False
    
    doppler_hz_list = np.array(doppler_hz_list)
    
    print(f"  ✓ Doppler samples: {len(doppler_hz_list)}")
    print(f"  ✓ Mean: {np.mean(doppler_hz_list):.2f} Hz")
    print(f"  ✓ Std: {np.std(doppler_hz_list):.2f} Hz")
    print(f"  ✓ Min: {np.min(doppler_hz_list):.2f} Hz")
    print(f"  ✓ Max: {np.max(doppler_hz_list):.2f} Hz")
    print(f"  ✓ Non-zero: {np.sum(np.abs(doppler_hz_list) > 1.0)}/{len(doppler_hz_list)} ({100*np.sum(np.abs(doppler_hz_list) > 1.0)/len(doppler_hz_list):.1f}%)")
    
    # Check for reasonable distribution (should not be all zeros or all same value)
    if np.std(doppler_hz_list) < 1.0:
        print(f"  ⚠️  WARNING: Doppler std too low ({np.std(doppler_hz_list):.2f} Hz) - may indicate no Doppler applied!")
    elif np.std(doppler_hz_list) > 1e6:
        print(f"  ⚠️  WARNING: Doppler std very high ({np.std(doppler_hz_list):.2f} Hz) - may indicate inconsistent calculation!")
    else:
        print(f"  ✓ Doppler distribution looks reasonable")
    
    # ========================================================================
    # CHECK 4: Meta Consistency (power_diff_pct)
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 CHECK 4: Meta Consistency (power_diff_pct)")
    print("="*70)
    
    power_diff_pct_list = []
    power_before_list = []
    power_after_list = []
    
    for m in dataset['meta']:
        if isinstance(m, dict):
            if 'power_diff_pct' in m:
                power_diff_pct_list.append(m['power_diff_pct'])
            if 'power_before' in m:
                power_before_list.append(m['power_before'])
            if 'power_after' in m:
                power_after_list.append(m['power_after'])
    
    if len(power_diff_pct_list) == 0:
        print("  ⚠️  WARNING: 'power_diff_pct' not found in meta (may be old dataset)")
        print("  → Re-generate dataset to include power_diff_pct in meta")
    else:
        power_diff_pct_list = np.array(power_diff_pct_list)
        print(f"  ✓ power_diff_pct samples: {len(power_diff_pct_list)}")
        print(f"  ✓ Mean: {np.mean(power_diff_pct_list):.2f}%")
        print(f"  ✓ Std: {np.std(power_diff_pct_list):.2f}%")
        print(f"  ✓ Min: {np.min(power_diff_pct_list):.2f}%")
        print(f"  ✓ Max: {np.max(power_diff_pct_list):.2f}%")
        
        # Check attack samples only
        attack_power_diffs = []
        for i, m in enumerate(dataset['meta']):
            if isinstance(m, dict) and 'power_diff_pct' in m and labels[i] == 1:
                attack_power_diffs.append(m['power_diff_pct'])
        
        if len(attack_power_diffs) > 0:
            attack_power_diffs = np.array(attack_power_diffs)
            print(f"\n  📊 Attack samples power_diff_pct:")
            print(f"      Mean: {np.mean(attack_power_diffs):.2f}%")
            print(f"      Std: {np.std(attack_power_diffs):.2f}%")
            print(f"      Range: [{np.min(attack_power_diffs):.2f}%, {np.max(attack_power_diffs):.2f}%]")
            
            if POWER_PRESERVING_COVERT:
                if np.mean(attack_power_diffs) > 10.0:
                    print(f"      ⚠️  WARNING: Mean power_diff_pct={np.mean(attack_power_diffs):.2f}% > 10% (power-preserving enabled!)")
                else:
                    print(f"      ✓ Power-preserving works (mean diff < 10%)")
            else:
                if np.mean(attack_power_diffs) < 1.0:
                    print(f"      ⚠️  WARNING: Mean power_diff_pct={np.mean(attack_power_diffs):.2f}% < 1% (injection may be too weak!)")
                elif np.mean(attack_power_diffs) < 5.0:
                    print(f"      ⚠️  WARNING: Mean power_diff_pct={np.mean(attack_power_diffs):.2f}% < 5% (injection may be hard to detect)")
                else:
                    print(f"      ✓ Power difference sufficient for detection")
    
    # ========================================================================
    # CHECK 5: Cross-GPU Consistency (if metadata available)
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 CHECK 5: Cross-GPU Consistency Check")
    print("="*70)
    
    # Split dataset into halves (assuming first half from GPU0, second from GPU1)
    if total_samples >= 100:
        mid_point = total_samples // 2
        first_half_labels = labels[:mid_point]
        second_half_labels = labels[mid_point:]
        
        first_half_benign = np.sum(first_half_labels == 0)
        first_half_attack = np.sum(first_half_labels == 1)
        second_half_benign = np.sum(second_half_labels == 0)
        second_half_attack = np.sum(second_half_labels == 1)
        
        print(f"  📊 First half (samples 0-{mid_point-1}):")
        print(f"      Benign: {first_half_benign}, Attack: {first_half_attack}")
        print(f"  📊 Second half (samples {mid_point}-{total_samples-1}):")
        print(f"      Benign: {second_half_benign}, Attack: {second_half_attack}")
        
        # Check if halves are balanced
        first_half_ratio = first_half_benign / (first_half_benign + first_half_attack + 1e-12)
        second_half_ratio = second_half_benign / (second_half_benign + second_half_attack + 1e-12)
        
        if abs(first_half_ratio - 0.5) > 0.2 or abs(second_half_ratio - 0.5) > 0.2:
            print(f"  ⚠️  WARNING: Halves may be imbalanced!")
            print(f"      First half benign ratio: {first_half_ratio:.2%}")
            print(f"      Second half benign ratio: {second_half_ratio:.2%}")
        else:
            print(f"  ✓ Halves appear balanced")
        
        # Check power consistency between halves
        first_half_powers = []
        second_half_powers = []
        
        for i in range(mid_point):
            grid = np.squeeze(tx_grids[i])
            first_half_powers.append(np.mean(np.abs(grid)**2))
        
        for i in range(mid_point, total_samples):
            grid = np.squeeze(tx_grids[i])
            second_half_powers.append(np.mean(np.abs(grid)**2))
        
        first_half_powers = np.array(first_half_powers)
        second_half_powers = np.array(second_half_powers)
        
        first_half_mean = np.mean(first_half_powers)
        second_half_mean = np.mean(second_half_powers)
        power_diff_halves = abs(first_half_mean - second_half_mean) / (first_half_mean + 1e-12) * 100.0
        
        print(f"\n  📊 Power consistency between halves:")
        print(f"      First half mean power: {first_half_mean:.6e}")
        print(f"      Second half mean power: {second_half_mean:.6e}")
        print(f"      Difference: {power_diff_halves:.2f}%")
        
        if power_diff_halves > 10.0:
            print(f"      ⚠️  WARNING: Large power difference between halves ({power_diff_halves:.2f}%)!")
            print(f"      → May indicate GPU-specific power scaling issues")
        else:
            print(f"      ✓ Power consistent between halves")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("📋 CONSISTENCY SUMMARY")
    print("="*70)
    
    all_checks_passed = True
    
    # Check if all critical checks passed
    if total_samples != expected_total:
        print("  ⚠️  WARNING: Sample count mismatch")
        all_checks_passed = False
    
    if benign_count != expected_benign or attack_count != expected_attack:
        print("  ⚠️  WARNING: Class imbalance detected")
        all_checks_passed = False
    
    if len(power_diff_pct_list) == 0:
        print("  ⚠️  WARNING: power_diff_pct missing in meta (re-generate dataset)")
        all_checks_passed = False
    
    if all_checks_passed:
        print("  ✅ All critical checks passed!")
    else:
        print("  ⚠️  Some warnings detected - review above")
    
    return all_checks_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dataset consistency after merge")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset file (default: auto-detect from DATASET_DIR)")
    
    args = parser.parse_args()
    
    if args.dataset:
        dataset_path = args.dataset
    else:
        # Auto-detect latest dataset
        dataset_dir = DATASET_DIR
        if not os.path.exists(dataset_dir):
            print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
            sys.exit(1)
        
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
        if not dataset_files:
            print(f"❌ ERROR: No .pkl files found in {dataset_dir}")
            sys.exit(1)
        
        # Use most recent file
        dataset_files.sort(key=lambda x: os.path.getmtime(os.path.join(dataset_dir, x)), reverse=True)
        dataset_path = os.path.join(dataset_dir, dataset_files[0])
        print(f"📂 Auto-detected dataset: {dataset_files[0]}")
    
    success = check_dataset_consistency(dataset_path)
    sys.exit(0 if success else 1)

