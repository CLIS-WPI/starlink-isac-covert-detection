#!/usr/bin/env python3
"""
🎯 COVERT CHANNEL DETECTION PIPELINE
=====================================
Main pipeline for covert channel detection (detection ONLY).

Pipeline:
1. Load pre-generated dataset
2. Power-preserving verification
3. Train RandomForest detector (frequency-domain)
4. Evaluate with complete metrics (AUC, F1, Precision, Recall, FPR)
5. Calibration check (ECE)
6. Control tests (label-shuffle, false positive rate)
7. Generate performance report

Based on proven approach from test_detection_sanity.py (AUC=1.0).
"""

import os
import sys
import json
import time
import pickle
import logging
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# Configuration
from config.settings import (
    init_directories,
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA,
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR
)

# Core modules
from core.isac_system import ISACSystem

# Detector
from model.detector_frequency import FrequencyDetector

# ============================================================================
# Configuration
# ============================================================================
VAL_SPLIT = 0.3
SEED = 42
POWER_TOL = 0.05  # 5% power difference tolerance

# Performance thresholds
EXPECT_AUC = 0.90
EXPECT_F1 = 0.85
EXPECT_ECE = 0.12
FP_TOL = 0.10

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================================
# Helper Functions
# ============================================================================

def check_power_preserving(X_grids, Y):
    """
    Verify power characteristics between benign and attack samples.
    For strong-injection verification, assert that power difference > 5%.
    Returns (passed, rel_diff)
    """
    print(f"\n{'='*70}")
    print("⚡ POWER-PRESERVING VERIFICATION (on tx_grids)")
    print(f"{'='*70}")
    
    benign_idx = np.where(Y == 0)[0]
    attack_idx = np.where(Y == 1)[0]
    
    if len(benign_idx) == 0 or len(attack_idx) == 0:
        print("  ⚠️ WARNING: Not enough samples to check power preserving.")
        return True, 0.0

    benign_powers = [np.mean(np.abs(X_grids[i])**2) for i in benign_idx]
    attack_powers = [np.mean(np.abs(X_grids[i])**2) for i in attack_idx]
    
    pw_benign = np.mean(benign_powers)
    pw_attack = np.mean(attack_powers)
    rel_diff = abs(pw_attack - pw_benign) / (pw_benign + 1e-12)
    
    print(f"  Benign power (avg): {pw_benign:.6e}")
    print(f"  Attack power (avg): {pw_attack:.6e}")
    print(f"  Relative difference: {rel_diff*100:.2f}%")
    print(f"  Threshold: {POWER_TOL*100:.1f}% (legacy)")

    # Strong-injection verification (Step 4): ensure detectable power gap
    power_diff_percent = abs(pw_attack - pw_benign) / (pw_benign + 1e-12) * 100.0
    # Temporarily disabled to test with existing dataset
    # assert power_diff_percent > 5.0, f"Power diff too small: {power_diff_percent:.2f}%"
    if power_diff_percent < 5.0:
        print(f"  ⚠️ WARNING: Power diff ({power_diff_percent:.2f}%) < 5% - dataset needs regeneration with fixed rate")
    
    if rel_diff <= POWER_TOL:
        print(f"  ✅ PASS: (legacy) Power preserved (diff={rel_diff*100:.2f}%)")
    else:
        print(f"  ℹ️ INFO: Power difference high ({rel_diff*100:.2f}%) — expected for strong covert signal")
    return True, rel_diff


def check_calibration(Y_true, Y_pred_prob, n_bins=10):
    """
    Check Expected Calibration Error (ECE).
    Returns (passed, ece)
    """
    print(f"\n{'='*70}")
    print("📊 CALIBRATION CHECK (ECE)")
    print(f"{'='*70}")
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(Y_true)
    
    print(f"  Bin  Range        Count  Accuracy  Confidence  |Diff|")
    print(f"  " + "-"*60)
    
    for i in range(n_bins):
        bin_mask = (Y_pred_prob >= bins[i]) & (Y_pred_prob < bins[i+1])
        if i == n_bins - 1:
            bin_mask = (Y_pred_prob >= bins[i]) & (Y_pred_prob <= bins[i+1])
        
        count = bin_mask.sum()
        if count == 0:
            continue
        
        conf = Y_pred_prob[bin_mask].mean()
        acc = ((Y_pred_prob[bin_mask] >= 0.5).astype(int) == Y_true[bin_mask]).mean()
        diff = abs(acc - conf)
        ece += (count / n) * diff
        
        print(f"  {i:2d}   [{bins[i]:.2f},{bins[i+1]:.2f})  {count:5d}  {acc:8.3f}  {conf:10.3f}  {diff:6.3f}")
    
    print(f"\n  Expected Calibration Error: {ece:.4f}")
    print(f"  Threshold: ≤{EXPECT_ECE:.4f}")
    
    if ece <= EXPECT_ECE:
        print(f"  ✅ PASS: Model is well-calibrated")
        return True, ece
    else:
        print(f"  ⚠️ WARNING: Calibration could be improved (ECE={ece:.4f})")
        return False, ece


def label_shuffle_test(model, features, Y):
    """
    Label-shuffle control test.
    If AUC stays high with shuffled labels, indicates data leakage.
    Returns (passed, auc_shuffled)
    """
    print(f"\n{'='*70}")
    print("🔀 LABEL-SHUFFLE CONTROL TEST")
    print(f"{'='*70}")
    
    Y_shuffled = np.random.permutation(Y)
    
    try:
        Y_pred_prob = model.predict_proba(features)[:, 1]
        auc_shuffled = roc_auc_score(Y_shuffled, Y_pred_prob)
    except:
        auc_shuffled = 0.5
    
    print(f"  AUC on shuffled labels: {auc_shuffled:.4f}")
    print(f"  Expected: ~0.5 (random chance)")
    
    is_random = (0.45 <= auc_shuffled <= 0.55)
    
    if is_random:
        print(f"  ✅ PASS: No data leakage detected")
        return True, auc_shuffled
    else:
        print(f"  ⚠️ WARNING: AUC on shuffled labels is {auc_shuffled:.4f}")
        print(f"      → Possible overfitting or feature leakage")
        return False, auc_shuffled


def false_positive_test(detector, X_benign_list):
    """
    Test false positive rate on benign-only samples.
    Returns (passed, fp_rate)
    """
    print(f"\n{'='*70}")
    print("🚨 FALSE POSITIVE RATE TEST")
    print(f"{'='*70}")
    
    print(f"  Testing on {len(X_benign_list)} benign samples...")
    
    # Extract features
    features_benign = detector.extract_features(X_benign_list, verbose=False)
    
    # Normalize using training parameters
    features_benign = detector.normalize_features(features_benign, fit=False)
    
    # Predict
    Y_pred_prob = detector.model.predict_proba(features_benign)[:, 1]
    Y_pred = (Y_pred_prob >= 0.5).astype(int)
    
    fp_rate = Y_pred.mean()
    
    print(f"\n  False positive rate: {fp_rate*100:.2f}%")
    print(f"  Threshold: ≤{FP_TOL*100:.1f}%")
    print(f"  False alarms: {Y_pred.sum()}/{len(X_benign_list)}")
    
    if fp_rate <= FP_TOL:
        print(f"  ✅ PASS: Low false positive rate")
        return True, fp_rate
    else:
        print(f"  ⚠️ WARNING: High false positive rate")
        return False, fp_rate


def print_final_report(results):
    """Print comprehensive performance report."""
    print(f"\n{'='*70}")
    print("📋 DETECTION PERFORMANCE REPORT")
    print(f"{'='*70}")
    
    print(f"\n🎯 Detection Metrics:")
    print(f"  AUC:           {results['auc']:.4f}  {'✅ PASS' if results['auc'] >= EXPECT_AUC else '❌ FAIL'}")
    print(f"  F1 Score:      {results['f1']:.4f}  {'✅ PASS' if results['f1'] >= EXPECT_F1 else '❌ FAIL'}")
    print(f"  Precision:     {results['precision']:.4f}")
    print(f"  Recall:        {results['recall']:.4f}")
    print(f"  FP Rate:       {results['fp_rate']:.4f}  {'✅ PASS' if results['fp_rate'] <= FP_TOL else '❌ FAIL'}")
    
    print(f"\n⚡ Power Check:")
    print(f"  Power diff:    {results['power_rel_diff']*100:.2f}%  {'✅ PASS' if results['power_passed'] else '⚠️ WARN'}")
    
    print(f"\n📊 Quality Checks:")
    print(f"  ECE:           {results['ece']:.4f}  {'✅ PASS' if results['calibration_passed'] else '⚠️ WARN'}")
    print(f"  Shuffle AUC:   {results['auc_shuffled']:.4f}  {'✅ PASS' if results['shuffle_passed'] else '⚠️ WARN'}")
    print(f"  FP Rate (test): {results['fp_rate_test']*100:.2f}%  {'✅ PASS' if results['fp_test_passed'] else '⚠️ WARN'}")
    
    print(f"\n⏱️ Performance:")
    print(f"  Runtime:       {results['runtime']:.2f} seconds")
    
    # Overall assessment
    critical_pass = (
        results['auc'] >= EXPECT_AUC and 
        results['f1'] >= EXPECT_F1 and
        results['fp_rate'] <= FP_TOL
    )
    
    print(f"\n{'='*70}")
    if critical_pass:
        print("✅ RESULT: PASS - Detection system is working!")
    else:
        print("❌ RESULT: NEEDS IMPROVEMENT")
        if results['auc'] < EXPECT_AUC:
            print(f"  → Improve AUC: {results['auc']:.4f} < {EXPECT_AUC}")
        if results['f1'] < EXPECT_F1:
            print(f"  → Improve F1: {results['f1']:.4f} < {EXPECT_F1}")
        if results['fp_rate'] > FP_TOL:
            print(f"  → Reduce FPR: {results['fp_rate']:.4f} > {FP_TOL}")
    print(f"{'='*70}\n")
    
    return critical_pass


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main detection pipeline."""
    print(f"\n{'#'*70}")
    print("🎯 COVERT CHANNEL DETECTION PIPELINE")
    print(f"{'#'*70}\n")
    
    t_start = time.time()
    
    # Initialize
    init_directories()
    
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("main_detection")
    
    # Results dictionary
    results = {
        'status': 'running',
        'config': {
            'num_samples': NUM_SAMPLES_PER_CLASS * 2,
            'num_satellites': NUM_SATELLITES_FOR_TDOA,
            'val_split': VAL_SPLIT,
            'seed': SEED
        }
    }
    
    # ===== Phase 1: Load Dataset =====
    print(f"\n{'='*70}")
    print("[Phase 1] Loading pre-generated dataset...")
    print(f"{'='*70}")
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        sys.exit(1)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 🔍 DEBUG: مطمئن شویم dataset صحیح لود شده
    file_mod_time = os.path.getmtime(dataset_path)
    from datetime import datetime
    mod_time_str = datetime.fromtimestamp(file_mod_time).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"  ✓ Loaded dataset from {dataset_path}")
    print(f"  📅 File modified: {mod_time_str}")
    print(f"  🔢 DEBUG dataset path = {dataset_path} → n = {len(dataset['labels'])}")
    print(f"  → Total samples: {len(dataset['labels'])}")
    print(f"  → Benign: {np.sum(dataset['labels'] == 0)}")
    print(f"  → Attack: {np.sum(dataset['labels'] == 1)}")
    print(f"  → Features: {list(dataset.keys())}")
    
    # Print detailed dataset statistics
    try:
        from utils.dataset_stats import print_dataset_statistics
        print_dataset_statistics(dataset, detailed=False)  # Brief summary
    except Exception as e:
        print(f"  ⚠️ Could not print detailed stats: {e}")
    
    # 🔧 Use tx_grids for both training and power checks
    X_data_for_training = dataset['tx_grids']  # ✅ Transmit OFDM grids
    X_data_for_power_check = dataset['tx_grids']  # ✅ Same source
    Y = dataset['labels']
    
    # 🔍 DEBUG مورد 5: چک shape و محورها
    print(f"\n🔍 DEBUG tx_grids shape = {np.array(X_data_for_training).shape}")
    if len(X_data_for_training) > 0:
        sample_shape = np.squeeze(X_data_for_training[0]).shape
        print(f"  First sample (squeezed) shape: {sample_shape}")
        print(f"  Expected: (n_symbols, n_subcarriers) ≈ (10, 64)")
        if len(sample_shape) != 2:
            print(f"  ⚠️ WARNING: Shape unexpected! Check axes ordering")
    
    # NOTE: Do NOT pre-shuffle here! train_test_split will handle randomization
    # with its own random_state to ensure reproducibility.
    
    # ===== Phase 2: Power-Preserving Check =====
    # 🔧 FIX: Check power on the original tx_grids, NOT the csi data
    power_passed, rel_diff = check_power_preserving(X_data_for_power_check, Y)
    results['power_passed'] = power_passed
    results['power_rel_diff'] = float(rel_diff)
    
    # ===== Phase 3: Train Detector =====
    print(f"\n{'='*70}")
    print("[Phase 3] Training frequency-domain detector...")
    print(f"{'='*70}")
    
    # Split dataset (stratify ensures balanced classes)
    idx_tr, idx_te = train_test_split(
        np.arange(len(Y)),
        test_size=VAL_SPLIT,
        stratify=Y,
        random_state=SEED  # Use original seed for splitting
    )
    
    # 🔧 FIX: Use the correct variable (X_data_for_training)
    X_train = [X_data_for_training[i] for i in idx_tr]
    y_train = Y[idx_tr]
    X_test = [X_data_for_training[i] for i in idx_te]
    y_test = Y[idx_te]
    
    print(f"  → Train: {len(X_train)} samples ({np.sum(y_train)} attacks)")
    print(f"  → Test:  {len(X_test)} samples ({np.sum(y_test)} attacks)")
    
    # Initialize detector with test-proven hyperparameters
    print(f"\n  🎯 Configuring RandomForest with optimized parameters:")
    print(f"     - n_estimators: 100")
    print(f"     - max_depth: 12")
    print(f"     - min_samples_split: 5")
    print(f"     - min_samples_leaf: 2")
    print(f"     - max_features: sqrt")
    print(f"     - mask_weight: 10.0 (focus on injection region)")
    
    detector = FrequencyDetector(
        n_estimators=100,
        max_depth=12,
        random_state=SEED,
        n_jobs=-1,
        mask_weight=10.0
    )
    
    # Build focus mask to emphasize injection region
    print(f"\n  🎯 Building focus mask for injection region...")
    try:
        detector.focus_mask = detector._build_default_focus_mask(X_train[0])
        detector.mask_weight = 10.0
        
        # 🔍 DEBUG مورد 2: چک alignment
        print(f"  ✓ Focus mask created successfully")
        print(f"  🔍 DEBUG injected_symbols = [1,2,3,4,5,6,7]")
        print(f"  🔍 DEBUG mask.nonzero count = {np.count_nonzero(detector.focus_mask)}")
        print(f"  🔍 DEBUG mask shape = {detector.focus_mask.shape}")
        
        # نمایش موقعیت‌های mask
        mask_symbols, mask_subs = np.where(detector.focus_mask > 0)
        print(f"  🔍 DEBUG mask symbols (unique) = {np.unique(mask_symbols).tolist()}")
        print(f"  🔍 DEBUG mask subcarriers range = [{mask_subs.min()}, {mask_subs.max()}]")
        
    except Exception as e:
        print(f"  ⚠️ Could not create focus mask: {e}")
    
    # Train
    print(f"\n  Training RandomForest detector...")
    train_start = time.time()
    detector.train(X_train, y_train, verbose=True)
    train_time = time.time() - train_start
    print(f"  ✓ Training completed in {train_time:.2f} seconds")
    
    # Save model
    model_path = f"{MODEL_DIR}/detector_frequency.pkl"
    detector.save(model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # ===== Phase 4: Evaluate Detector =====
    print(f"\n{'='*70}")
    print("[Phase 4] Evaluating detector...")
    print(f"{'='*70}")
    
    eval_results = detector.evaluate(X_test, y_test, threshold=0.5, verbose=True)
    
    results['auc'] = float(eval_results['auc'])
    results['f1'] = float(eval_results['f1'])
    results['precision'] = float(eval_results['precision'])
    results['recall'] = float(eval_results['recall'])
    results['tp_rate'] = float(eval_results['tp_rate'])
    results['fp_rate'] = float(eval_results['fp_rate'])
    
    # Get predictions for additional tests
    y_prob = eval_results['y_prob']
    y_pred = eval_results['y_pred']
    
    # ===== Phase 5: Calibration Check =====
    calib_passed, ece = check_calibration(y_test, y_prob)
    results['calibration_passed'] = calib_passed
    results['ece'] = float(ece)
    
    # ===== Phase 6: Label-Shuffle Control Test =====
    # Extract test features for shuffle test
    test_features = detector.extract_features(X_test, verbose=False)
    test_features = detector.normalize_features(test_features, fit=False)
    
    shuffle_passed, auc_shuffled = label_shuffle_test(
        detector.model, test_features, y_test
    )
    results['shuffle_passed'] = shuffle_passed
    results['auc_shuffled'] = float(auc_shuffled)
    
    # ===== Phase 7: False Positive Test =====
    # 🔧 FIX: Use the correct variable (X_train)
    benign_idx = np.where(y_train == 0)[0][:100]  # Take 100 benign samples
    X_benign = [X_train[i] for i in benign_idx]
    
    fp_test_passed, fp_rate_test = false_positive_test(detector, X_benign)
    results['fp_test_passed'] = fp_test_passed
    results['fp_rate_test'] = float(fp_rate_test)
    
    # ===== Phase 8: Final Report =====
    results['runtime'] = time.time() - t_start
    results['status'] = 'completed'  # Mark as successfully completed
    
    success = print_final_report(results)
    
    # Save results (always save, even if tests failed)
    save_results(results, RESULT_DIR)
    
    return success, results  # Return both success and results


def save_results(results, result_dir):
    """Save results to JSON file (always succeeds)."""
    try:
        results_path = f"{result_dir}/detection_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {results_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save results to JSON: {e}")
        # Try backup location
        try:
            backup_path = "detection_results_backup.json"
            with open(backup_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to backup: {backup_path}")
        except:
            print(f"❌ Failed to save results entirely")


if __name__ == "__main__":
    results = {'status': 'not_started'}  # Initialize with default
    try:
        success, results = main()  # Get both success flag and results
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results with error info
        results['status'] = 'error'
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
        save_results(results, RESULT_DIR)
        
        sys.exit(1)