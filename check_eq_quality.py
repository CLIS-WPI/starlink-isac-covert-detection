#!/usr/bin/env python3
"""
🔍 Check EQ Quality Before Training
====================================
Validates that MMSE equalization is working correctly before CNN training.
"""

import pickle
import numpy as np
import glob
import sys

def check_eq_quality(dataset_path=None):
    """Check EQ quality metrics from dataset metadata."""
    
    # Find latest dataset if not provided
    if dataset_path is None:
        dataset_files = glob.glob('dataset/dataset_scenario_b_*.pkl')
        if not dataset_files:
            print("❌ No Scenario B dataset found!")
            return False
        dataset_path = max(dataset_files, key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
        print(f"📁 Using dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print("="*70)
    print("🔍 EQ Quality Check (Pre-Training Validation)")
    print("="*70)
    
    # Extract metadata
    meta_list = dataset.get('meta', [])
    if not meta_list:
        print("❌ No metadata found in dataset!")
        return False
    
    # Check EQ metrics
    snr_improvements = []
    preservation_values = []
    alpha_values = []
    blend_factors = []
    snr_raw_values = []
    snr_eq_values = []
    
    missing_eq = 0
    total_samples = len(meta_list)
    
    for i, meta in enumerate(meta_list):
        if isinstance(meta, tuple):
            _, meta = meta
        
        # Check if EQ info exists
        if 'eq_snr_improvement_db' not in meta:
            missing_eq += 1
            continue
        
        snr_imp = meta.get('eq_snr_improvement_db')
        snr_raw = meta.get('eq_snr_raw_db')
        snr_eq = meta.get('eq_snr_db')
        alpha = meta.get('eq_alpha')
        blend = meta.get('eq_blend_factor')
        
        if snr_imp is not None:
            snr_improvements.append(snr_imp)
        if snr_raw is not None:
            snr_raw_values.append(snr_raw)
        if snr_eq is not None:
            snr_eq_values.append(snr_eq)
        if alpha is not None:
            alpha_values.append(alpha)
        if blend is not None:
            blend_factors.append(blend)
        
        # Check for pattern preservation (if available)
        if 'eq_pattern_preservation' in meta:
            preservation_values.append(meta['eq_pattern_preservation'])
    
    print(f"\n📊 Sample Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with EQ info: {total_samples - missing_eq}")
    print(f"  Samples missing EQ info: {missing_eq}")
    
    if missing_eq > total_samples * 0.1:  # More than 10% missing
        print(f"  ⚠️  WARNING: {missing_eq} samples missing EQ info (>10%)!")
        return False
    
    if not snr_improvements:
        print("  ❌ No EQ metrics found!")
        return False
    
    # Compute statistics
    snr_imp_mean = np.mean(snr_improvements)
    snr_imp_median = np.median(snr_improvements)
    snr_imp_std = np.std(snr_improvements)
    
    print(f"\n📊 SNR Improvement Statistics:")
    print(f"  Mean: {snr_imp_mean:.2f} dB")
    print(f"  Median: {snr_imp_median:.2f} dB")
    print(f"  Std: {snr_imp_std:.2f} dB")
    print(f"  Min: {np.min(snr_improvements):.2f} dB")
    print(f"  Max: {np.max(snr_improvements):.2f} dB")
    
    if snr_raw_values:
        print(f"\n📊 SNR Raw Statistics:")
        print(f"  Mean: {np.mean(snr_raw_values):.2f} dB")
        print(f"  Median: {np.median(snr_raw_values):.2f} dB")
    
    if snr_eq_values:
        print(f"\n📊 SNR EQ Statistics:")
        print(f"  Mean: {np.mean(snr_eq_values):.2f} dB")
        print(f"  Median: {np.median(snr_eq_values):.2f} dB")
    
    if preservation_values:
        pres_mean = np.mean(preservation_values)
        pres_median = np.median(preservation_values)
        print(f"\n📊 Pattern Preservation Statistics:")
        print(f"  Mean: {pres_mean:.3f}")
        print(f"  Median: {pres_median:.3f}")
    
    if alpha_values:
        print(f"\n📊 Alpha Statistics:")
        print(f"  Mean: {np.mean(alpha_values):.6e}")
        print(f"  Median: {np.median(alpha_values):.6e}")
    
    if blend_factors:
        print(f"\n📊 Blend Factor Statistics:")
        print(f"  Mean: {np.mean(blend_factors):.3f}")
        print(f"  Median: {np.median(blend_factors):.3f}")
        print(f"  Samples with full EQ (blend=1.0): {np.sum(np.array(blend_factors) == 1.0)}")
        print(f"  Samples with partial EQ (blend<1.0): {np.sum(np.array(blend_factors) < 1.0)}")
    
    # Acceptance criteria
    print(f"\n✅ Acceptance Criteria:")
    print(f"  Mean SNR improvement ≥ 4 dB: {snr_imp_mean >= 4.0} ({snr_imp_mean:.2f} dB)")
    if preservation_values:
        pres_median = np.median(preservation_values)
        print(f"  Median preservation ≥ 0.5: {pres_median >= 0.5} ({pres_median:.3f})")
    else:
        print(f"  Median preservation ≥ 0.5: ⚠️  (not available)")
    
    # Final verdict
    passed = True
    if snr_imp_mean < 4.0:
        print(f"  ❌ FAILED: Mean SNR improvement < 4 dB")
        passed = False
    else:
        print(f"  ✅ PASSED: Mean SNR improvement ≥ 4 dB")
    
    if preservation_values:
        if np.median(preservation_values) < 0.5:
            print(f"  ❌ FAILED: Median preservation < 0.5")
            passed = False
        else:
            print(f"  ✅ PASSED: Median preservation ≥ 0.5")
    
    print(f"\n{'='*70}")
    if passed:
        print("✅ EQ Quality Check: PASSED")
        print("  → Ready for CNN training")
    else:
        print("❌ EQ Quality Check: FAILED")
        print("  → EQ needs tuning before training")
    print(f"{'='*70}\n")
    
    return passed

if __name__ == '__main__':
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = check_eq_quality(dataset_path)
    sys.exit(0 if success else 1)

