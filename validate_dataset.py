#!/usr/bin/env python3
"""
🔍 Dataset Validation Script (Phase 0 Enhanced)
===============================================
Enhanced validation with normalization leakage detection and CSV export.

Phase 0: Infrastructure hardening for evaluation.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA,
    INSIDER_MODE, POWER_PRESERVING_COVERT, COVERT_AMP, CARRIER_FREQUENCY,
    GLOBAL_SEED, RESULT_DIR
)

# 🔒 Phase 0: Set global seeds
from utils.reproducibility import set_global_seeds, log_seed_info
log_seed_info("validate_dataset.py")
set_global_seeds(deterministic=True)


def check_normalization_leakage(dataset, output_csv=None):
    """
    Check for data leakage in normalization (Phase 0).
    
    Computes normalization statistics (mean/std) on train-only data,
    then checks if val/test statistics differ significantly.
    
    Args:
        dataset: Dataset dictionary
        output_csv: Path to save validation CSV
    
    Returns:
        dict: Validation results
    """
    print("\n" + "="*70)
    print("🔍 CHECK 6: Normalization Leakage Detection (Phase 0)")
    print("="*70)
    
    if 'tx_grids' not in dataset or 'labels' not in dataset:
        print("  ❌ FAIL: Missing tx_grids or labels")
        return None
    
    # Split data (same as training pipeline)
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    # 70/20/10 split (same as main_detection_cnn.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X_grids, Y,
        test_size=0.3,
        random_state=GLOBAL_SEED,
        stratify=Y
    )
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2/0.7,  # 20% of total = 20/70 of train
        random_state=GLOBAL_SEED,
        stratify=y_train
    )
    
    print(f"  ✓ Data split: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Compute normalization statistics on TRAIN ONLY
    # Convert to magnitude for normalization check
    X_tr_mag = np.abs(X_tr)
    train_mean = float(np.mean(X_tr_mag))
    train_std = float(np.std(X_tr_mag))
    
    print(f"  ✓ Train-only statistics:")
    print(f"      Mean: {train_mean:.6e}")
    print(f"      Std:  {train_std:.6e}")
    
    # Check validation and test statistics
    X_val_mag = np.abs(X_val)
    X_test_mag = np.abs(X_test)
    
    val_mean = float(np.mean(X_val_mag))
    val_std = float(np.std(X_val_mag))
    test_mean = float(np.mean(X_test_mag))
    test_std = float(np.std(X_test_mag))
    
    print(f"  ✓ Validation statistics:")
    print(f"      Mean: {val_mean:.6e} (diff from train: {abs(val_mean - train_mean)/train_mean*100:.2f}%)")
    print(f"      Std:  {val_std:.6e} (diff from train: {abs(val_std - train_std)/train_std*100:.2f}%)")
    
    print(f"  ✓ Test statistics:")
    print(f"      Mean: {test_mean:.6e} (diff from train: {abs(test_mean - train_mean)/train_mean*100:.2f}%)")
    print(f"      Std:  {test_std:.6e} (diff from train: {abs(test_std - train_std)/train_std*100:.2f}%)")
    
    # Check for significant differences (leakage indicator)
    mean_diff_val = abs(val_mean - train_mean) / train_mean * 100
    std_diff_val = abs(val_std - train_std) / train_std * 100
    mean_diff_test = abs(test_mean - train_mean) / train_mean * 100
    std_diff_test = abs(test_std - train_std) / train_std * 100
    
    # Threshold: if difference > 5%, may indicate leakage
    leakage_detected = False
    if mean_diff_val > 5.0 or std_diff_val > 5.0:
        print(f"  ⚠️  WARNING: Large difference in validation statistics (may indicate leakage)")
        leakage_detected = True
    if mean_diff_test > 5.0 or std_diff_test > 5.0:
        print(f"  ⚠️  WARNING: Large difference in test statistics (may indicate leakage)")
        leakage_detected = True
    
    if not leakage_detected:
        print(f"  ✅ PASS: No normalization leakage detected (differences < 5%)")
    
    # Prepare results for CSV export
    results = {
        'check_name': 'normalization_leakage',
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'mean_diff_val_pct': mean_diff_val,
        'std_diff_val_pct': std_diff_val,
        'mean_diff_test_pct': mean_diff_test,
        'std_diff_test_pct': std_diff_test,
        'leakage_detected': leakage_detected,
        'pass': not leakage_detected
    }
    
    return results


def check_power_analysis(dataset):
    """Compute power analysis for CSV export."""
    if 'tx_grids' not in dataset or 'labels' not in dataset:
        return None
    
    labels = dataset['labels']
    benign_mask = (labels == 0)
    attack_mask = (labels == 1)
    
    benign_powers = [np.mean(np.abs(dataset['tx_grids'][i])**2) for i in np.where(benign_mask)[0]]
    attack_powers = [np.mean(np.abs(dataset['tx_grids'][i])**2) for i in np.where(attack_mask)[0]]
    
    benign_power_mean = float(np.mean(benign_powers))
    attack_power_mean = float(np.mean(attack_powers))
    power_diff_pct = abs(attack_power_mean - benign_power_mean) / (benign_power_mean + 1e-12) * 100.0
    
    return {
        'benign_power_mean': benign_power_mean,
        'attack_power_mean': attack_power_mean,
        'power_diff_pct': power_diff_pct
    }


def check_csi_quality(dataset):
    """Compute CSI quality metrics for CSV export."""
    if 'csi_est' not in dataset or dataset['csi_est'] is None:
        return None
    
    csi_est = dataset['csi_est']
    csi_mag = np.abs(csi_est)
    csi_variance = float(np.var(csi_mag))
    csi_mean = float(np.mean(csi_mag))
    csi_std = float(np.std(csi_mag))
    
    # Compute NMSE if true CSI is available (placeholder for now)
    nmse_db = None  # Will be computed if H_true is available
    
    return {
        'csi_mean': csi_mean,
        'csi_std': csi_std,
        'csi_variance': csi_variance,
        'csi_nmse_db': nmse_db
    }


def check_doppler(dataset):
    """Check Doppler statistics."""
    if 'meta' not in dataset or not dataset['meta']:
        return None
    
    dopplers = []
    for m in dataset['meta']:
        if isinstance(m, dict) and 'doppler_hz' in m:
            dopplers.append(m['doppler_hz'])
    
    if len(dopplers) == 0:
        return None
    
    dopplers = np.array(dopplers)
    return {
        'doppler_mean_hz': float(np.mean(dopplers)),
        'doppler_std_hz': float(np.std(dopplers)),
        'doppler_min_hz': float(np.min(dopplers)),
        'doppler_max_hz': float(np.max(dopplers))
    }


def main():
    """Run all validation checks with CSV export."""
    parser = argparse.ArgumentParser(description="Dataset validation with Phase 0 enhancements")
    parser.add_argument('--dataset', type=str, default=None, help="Path to dataset file")
    parser.add_argument('--output-csv', type=str, default=None, help="Path to output CSV file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 DATASET VALIDATION CHECKLIST (Phase 0 Enhanced)")
    print("="*70)
    print(f"Scenario: {INSIDER_MODE}")
    print(f"Power-preserving: {POWER_PRESERVING_COVERT}")
    print(f"COVERT_AMP: {COVERT_AMP}")
    print(f"Global Seed: {GLOBAL_SEED}")
    print("="*70)
    
    # 🔧 FIX: Auto-detect latest dataset if not provided or if provided path doesn't exist
    if args.dataset is None or not os.path.exists(args.dataset):
        import glob
        
        # Determine scenario name (INSIDER_MODE already imported at top)
        scenario_name = 'scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b'
        
        # Find latest scenario-specific dataset
        dataset_files = glob.glob(os.path.join(DATASET_DIR, f"dataset_{scenario_name}*.pkl"))
        if dataset_files:
            # Sort by modification time (newest first)
            dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            dataset_path = dataset_files[0]  # Latest/newest
            if args.dataset is None:
                print(f"  → Auto-detected latest dataset: {os.path.basename(dataset_path)}")
            else:
                print(f"  ⚠️  Provided dataset not found, using latest: {os.path.basename(dataset_path)}")
        else:
            # Fallback
            dataset_path = (
                f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
                f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
            )
            if args.dataset is None:
                print(f"  → Using fallback: {dataset_path}")
    else:
        dataset_path = args.dataset
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        return False
    
    print(f"\n📂 Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Dataset loaded: {len(dataset.get('labels', []))} samples")
    
    # Run validation checks
    validation_results = {}
    
    # Check 1: Normalization leakage (Phase 0)
    norm_results = check_normalization_leakage(dataset)
    if norm_results:
        validation_results.update(norm_results)
    
    # Check 2: Power analysis
    power_results = check_power_analysis(dataset)
    if power_results:
        validation_results.update(power_results)
        print(f"\n  ✓ Power diff: {power_results['power_diff_pct']:.2f}%")
    
    # Check 3: CSI quality
    csi_results = check_csi_quality(dataset)
    if csi_results:
        validation_results.update(csi_results)
        print(f"\n  ✓ CSI variance: {csi_results['csi_variance']:.6e}")
    
    # Check 4: Doppler statistics
    doppler_results = check_doppler(dataset)
    if doppler_results:
        validation_results.update(doppler_results)
        print(f"\n  ✓ Doppler: mean={doppler_results['doppler_mean_hz']:.2f} Hz, std={doppler_results['doppler_std_hz']:.2f} Hz")
    
    # Add metadata
    validation_results['dataset_path'] = dataset_path
    validation_results['num_samples'] = len(dataset.get('labels', []))
    validation_results['insider_mode'] = INSIDER_MODE
    validation_results['power_preserving'] = POWER_PRESERVING_COVERT
    validation_results['covert_amp'] = COVERT_AMP
    validation_results['global_seed'] = GLOBAL_SEED
    
    # Export to CSV
    output_csv = args.output_csv or f"{RESULT_DIR}/validation_sanity.csv"
    try:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        df = pd.DataFrame([validation_results])
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Validation results exported to: {output_csv}")
    except PermissionError:
        # Fallback: save to current directory
        output_csv_fallback = "validation_sanity.csv"
        df = pd.DataFrame([validation_results])
        df.to_csv(output_csv_fallback, index=False)
        print(f"\n⚠️  Permission denied for {output_csv}, saved to: {output_csv_fallback}")
    except Exception as e:
        print(f"\n⚠️  Could not export CSV: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70)
    
    if norm_results and not norm_results.get('leakage_detected', False):
        print("  ✅ PASS: No normalization leakage")
    elif norm_results:
        print("  ⚠️  WARNING: Potential normalization leakage detected")
    
    if power_results and power_results['power_diff_pct'] < 5.0:
        print(f"  ✅ PASS: Power-preserving works (diff={power_results['power_diff_pct']:.2f}%)")
    elif power_results:
        print(f"  ⚠️  WARNING: Power diff high ({power_results['power_diff_pct']:.2f}%)")
    
    print(f"\n✅ Validation complete! Results saved to: {output_csv}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
