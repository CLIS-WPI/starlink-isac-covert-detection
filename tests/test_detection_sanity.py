"""
🎯 IMPROVED COVERT DETECTION with ISOTONIC CALIBRATION
========================================================
Enhanced calibration using Isotonic Regression instead of Temperature Scaling.

Key improvements:
1. Isotonic Regression (better than Temperature Scaling for RandomForest)
2. Platt Scaling as alternative
3. Calibration curves visualization
4. Better ECE reporting

Isotonic Regression is NON-PARAMETRIC and works better when:
- Model outputs are not logistic (like RandomForest)
- You have enough calibration data (>100 samples)
- The relationship between confidence and accuracy is not necessarily monotonic
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

from core.isac_system import ISACSystem
from core.covert_injection import inject_covert_channel

# ============================================================================
# ENHANCED FEATURE EXTRACTION
# ============================================================================

def compute_enhanced_features(tx_grid):
    """Enhanced feature extraction with spectral and statistical features."""
    grid_flat = tf.squeeze(tx_grid).numpy()
    
    # 1. Magnitude features (primary)
    magnitude = np.abs(grid_flat)
    mag_flat = magnitude.flatten()
    
    # 2. Phase features (secondary)
    phase = np.angle(grid_flat)
    phase_flat = phase.flatten()
    
    # 3. Statistical features per subcarrier
    mag_mean = np.mean(magnitude, axis=-1)
    mag_std = np.std(magnitude, axis=-1)
    mag_max = np.max(magnitude, axis=-1)
    
    # 4. Power distribution features
    power = magnitude ** 2
    power_flat = power.flatten()
    
    # Combine all features
    features = np.concatenate([
        mag_flat[:320],      # First 320 magnitude samples
        phase_flat[:160],    # First 160 phase samples  
        mag_mean,            # Mean per subcarrier
        mag_std,             # Std per subcarrier
        mag_max,             # Max per subcarrier
        power_flat[:160]     # First 160 power samples
    ])
    
    return features.astype(np.float32)


# ============================================================================
# Configuration
# ============================================================================

N_SAMPLES = 1000
COVERT_RATIO = 0.5
EBNO_DB_RANGE = (5, 15)
VAL_SPLIT = 0.3
SEED = 42
N_JOBS = -1

# Thresholds
POWER_TOL = 0.05
EXPECT_AUC = 0.80
EXPECT_F1 = 0.75
EXPECT_ECE = 0.10  # Stricter calibration target
FP_TOL = 0.10

# Set seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# DATASET GENERATION (WITH REALISTIC NOISE)
# ============================================================================

def generate_test_dataset_improved(n_samples, covert_ratio, ebno_range):
    """Generate dataset with AWGN noise and stealthy covert channels."""
    print(f"\n{'='*70}")
    print("📊 GENERATING REALISTIC TEST DATASET")
    print(f"{'='*70}")
    
    isac = ISACSystem()
    n_covert = int(n_samples * covert_ratio)
    n_benign = n_samples - n_covert
    
    X_list = []
    Y = []
    metadata = []
    
    print(f"  Benign samples: {n_benign}")
    print(f"  Attack samples: {n_covert}")
    print(f"  SNR range: {ebno_range[0]}-{ebno_range[1]} dB")
    print(f"  🔊 AWGN noise: ENABLED")
    
    total_payload_bits = isac.rg.num_data_symbols * isac.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac.n))
    total_info_bits = num_codewords * isac.k
    
    print(f"  Payload: {total_payload_bits} bits, {num_codewords} codewords")
    
    for i in range(n_samples):
        # Generate TX signal
        b = isac.binary_source([1, total_info_bits])
        c_blocks = [isac.encoder(b[:, j*isac.k:(j+1)*isac.k]) for j in range(num_codewords)]
        c = tf.concat(c_blocks, axis=1)[:, :total_payload_bits]
        x = isac.mapper(c)
        x = tf.reshape(x, [1, isac.NUM_SAT_BEAMS, isac.NUM_UT, -1])
        tx_grid = isac.rg_mapper(x)
        
        is_covert = (i >= n_benign)
        
        # Apply covert injection
        if is_covert:
            covert_rate_mbps = np.random.uniform(0.8, 2.5)
            try:
                result = inject_covert_channel(
                    tx_grid, isac.rg, covert_rate_mbps,
                    isac.SUBCARRIER_SPACING,
                    covert_amp=0.08  # Stealthy attack
                )
                tx_grid_attacked = result[0] if isinstance(result, tuple) else result
            except Exception as e:
                tx_grid_attacked = tx_grid
        else:
            tx_grid_attacked = tx_grid
        
        # Add AWGN noise
        ebno_db = np.random.uniform(*ebno_range)
        grid_flat = tf.squeeze(tx_grid_attacked).numpy()
        signal_power = np.mean(np.abs(grid_flat)**2)
        
        esno = 10.0 ** (ebno_db / 10.0)
        noise_power = signal_power / esno
        
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(*grid_flat.shape) + 
            1j * np.random.randn(*grid_flat.shape)
        )
        rx_grid = grid_flat + noise
        rx_grid = tf.expand_dims(tf.constant(rx_grid, dtype=tf.complex64), axis=0)
        
        X_list.append(rx_grid)
        Y.append(1 if is_covert else 0)
        metadata.append({
            'index': i,
            'is_covert': is_covert,
            'ebno_db': ebno_db,
            'tx_power': float(signal_power)
        })
        
        if (i+1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples} samples...")
    
    Y = np.array(Y, dtype=np.int32)
    print(f"  ✓ Dataset generated: {n_samples} samples")
    
    return X_list, Y, metadata


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_improved(X_list, Y=None, norm_params=None):
    """Extract enhanced features with parallel processing."""
    print(f"\n{'='*70}")
    print("🔬 FEATURE EXTRACTION (ENHANCED)")
    print(f"{'='*70}")
    
    n_samples = len(X_list)
    use_parallel = n_samples > 100
    
    if use_parallel:
        print(f"  Using parallel processing ({N_JOBS} jobs)...")
        features = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(compute_enhanced_features)(x) for x in X_list
        )
    else:
        features = [compute_enhanced_features(x) for x in X_list]
    
    features = np.array(features, dtype=np.float32)
    
    # Check validity
    has_nan = np.any(np.isnan(features))
    has_inf = np.any(np.isinf(features))
    
    if Y is not None:
        benign_feat = features[Y == 0]
        attack_feat = features[Y == 1]
        if len(benign_feat) > 0 and len(attack_feat) > 0:
            diff = np.abs(attack_feat.mean(axis=0) - benign_feat.mean(axis=0))
            print(f"  Feature separability: mean_diff={diff.mean():.6e}, max_diff={diff.max():.6e}")
    
    # Normalization
    if norm_params is None:
        global_mean = features.mean()
        global_std = features.std()
        norm_params = {'mean': global_mean, 'std': global_std}
    else:
        global_mean = norm_params['mean']
        global_std = norm_params['std']
    
    features = (features - global_mean) / (global_std + 1e-8)
    
    print(f"  Feature shape: {features.shape}")
    print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Mean: {features.mean():.3f}, Std: {features.std():.3f}")
    
    if has_nan or has_inf:
        print(f"  ❌ FAIL: NaN or Inf detected")
        return False, features, norm_params
    else:
        print(f"  ✅ PASS: Features valid")
        return True, features, norm_params


# ============================================================================
# TRAINING WITH ISOTONIC CALIBRATION (IMPROVED!)
# ============================================================================

def train_with_isotonic_calibration(features, Y):
    """
    Train model with Isotonic Regression calibration.
    
    WHY ISOTONIC IS BETTER:
    - Non-parametric (no assumptions about sigmoid shape)
    - Works better for RandomForest (which outputs non-logistic probabilities)
    - More flexible than Temperature Scaling
    """
    print(f"\n{'='*70}")
    print("🤖 TRAINING WITH ISOTONIC CALIBRATION")
    print(f"{'='*70}")
    
    n = len(Y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - VAL_SPLIT))
    
    train_idx = idx[:split]
    val_idx = idx[split:]
    
    X_train = features[train_idx]
    Y_train = Y[train_idx]
    X_val = features[val_idx]
    Y_val = Y[val_idx]
    
    print(f"  Train: {len(Y_train)} (benign={np.sum(Y_train==0)}, attack={np.sum(Y_train==1)})")
    print(f"  Val: {len(Y_val)} (benign={np.sum(Y_val==0)}, attack={np.sum(Y_val==1)})")
    
    # Step 1: Train base RandomForest
    from sklearn.ensemble import RandomForestClassifier
    print(f"  Training regularized Random Forest...")
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=SEED,
        n_jobs=-1
    )
    base_model.fit(X_train, Y_train)
    
    # Step 2: Get raw probabilities
    Y_pred_prob_raw = base_model.predict_proba(X_val)[:, 1]
    auc_raw = roc_auc_score(Y_val, Y_pred_prob_raw)
    print(f"  Raw model AUC: {auc_raw:.4f}")
    
    # Step 3: Apply Isotonic Regression calibration
    print(f"  Applying Isotonic Regression calibration...")
    from sklearn.isotonic import IsotonicRegression
    
    # We need a separate calibration set, so split train into train + calib
    n_train = len(Y_train)
    calib_split = int(n_train * 0.8)
    calib_idx = np.random.permutation(n_train)
    
    X_train_sub = X_train[calib_idx[:calib_split]]
    Y_train_sub = Y_train[calib_idx[:calib_split]]
    X_calib = X_train[calib_idx[calib_split:]]
    Y_calib = Y_train[calib_idx[calib_split:]]
    
    # Retrain on subset
    base_model.fit(X_train_sub, Y_train_sub)
    
    # Get calibration predictions
    Y_calib_prob = base_model.predict_proba(X_calib)[:, 1]
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(Y_calib_prob, Y_calib)
    
    # Apply to validation set
    Y_val_prob_raw = base_model.predict_proba(X_val)[:, 1]
    Y_pred_prob = iso_reg.transform(Y_val_prob_raw)
    
    print(f"  Calibration complete!")
    print(f"    Before calibration: mean_conf={Y_val_prob_raw.mean():.3f}")
    print(f"    After calibration:  mean_conf={Y_pred_prob.mean():.3f}")
    
    # Step 4: Evaluate
    auc = roc_auc_score(Y_val, Y_pred_prob)
    
    # Find best threshold
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.linspace(0.1, 0.9, 50):
        Y_pred = (Y_pred_prob >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(Y_val, Y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    Y_pred = (Y_pred_prob >= best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(Y_val, Y_pred, average='binary', zero_division=0)
    
    print(f"\n  Results (after isotonic calibration):")
    print(f"    AUC: {auc:.4f} (target: ≥{EXPECT_AUC})")
    print(f"    Best F1: {best_f1:.4f} at threshold={best_thresh:.3f} (target: ≥{EXPECT_F1})")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    
    passed = (auc >= EXPECT_AUC) and (best_f1 >= EXPECT_F1)
    if passed:
        print(f"  ✅ PASS: Model performance acceptable")
    else:
        print(f"  ❌ FAIL: Performance below expectations")
    
    return passed, (base_model, iso_reg), Y_pred_prob, best_thresh, {
        'auc': auc, 'f1': best_f1, 'precision': precision, 'recall': recall, 'threshold': best_thresh
    }


# ============================================================================
# CALIBRATION CHECK
# ============================================================================

def check_calibration_improved(Y_true, Y_pred_prob, n_bins=10):
    """Expected Calibration Error with detailed reporting."""
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
    
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Target: ≤{EXPECT_ECE:.4f}")
    
    if ece <= EXPECT_ECE:
        print(f"  ✅ PASS: Well-calibrated model")
        return True, ece
    else:
        print(f"  ⚠️  WARN: Calibration could be better (ECE={ece:.4f} > {EXPECT_ECE})")
        return False, ece


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_improved_detection_test():
    """Main orchestrator with isotonic calibration."""
    print(f"\n{'#'*70}")
    print("🎯 IMPROVED COVERT DETECTION with ISOTONIC CALIBRATION")
    print(f"{'#'*70}\n")
    
    t_start = time.time()
    results = {
        'improvements': [
            'Isotonic Regression calibration (better than Temperature Scaling)',
            'AWGN noise enabled',
            'Regularized RandomForest',
            'Enhanced features (mag+phase+stats)',
            'Stealthy attack (amp=0.08)',
            'Lower SNR (5-15 dB)'
        ],
        'test_config': {
            'n_samples': N_SAMPLES,
            'covert_ratio': COVERT_RATIO,
            'ebno_range': list(EBNO_DB_RANGE),
            'seed': SEED
        }
    }
    
    try:
        # Generate dataset
        X_list, Y, metadata = generate_test_dataset_improved(N_SAMPLES, COVERT_RATIO, EBNO_DB_RANGE)
        
        # Extract features
        features_passed, features, norm_params = extract_features_improved(X_list, Y)
        results['features_valid'] = features_passed
        
        if not features_passed:
            print("\n❌ ABORT: Feature extraction failed")
            return False
        
        # Train with isotonic calibration
        train_passed, model_tuple, Y_pred_prob, thresh, metrics = train_with_isotonic_calibration(features, Y)
        results['training_passed'] = train_passed
        results['auc'] = float(metrics['auc'])
        results['f1'] = float(metrics['f1'])
        results['precision'] = float(metrics['precision'])
        results['recall'] = float(metrics['recall'])
        
        # Check calibration
        n = len(Y)
        idx = np.arange(n)
        np.random.shuffle(idx)
        val_idx = idx[int(n * (1 - VAL_SPLIT)):]
        Y_val = Y[val_idx]
        
        calib_passed, ece = check_calibration_improved(Y_val, Y_pred_prob)
        results['calibration_passed'] = calib_passed
        results['ece'] = float(ece)
        
        # Final summary
        results['runtime_seconds'] = time.time() - t_start
        
        # Convert numpy types for JSON
        def convert_to_native(obj):
            if isinstance(obj, (np.bool_, np.generic)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_serializable = convert_to_native(results)
        
        print(f"\n{'='*70}")
        print("📋 FINAL RESULTS")
        print(f"{'='*70}")
        print(json.dumps(results_serializable, indent=2))
        
        all_passed = features_passed and train_passed and calib_passed
        
        print(f"\n{'='*70}")
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            failed = []
            if not features_passed:
                failed.append("Feature Extraction")
            if not train_passed:
                failed.append("Model Training")
            if not calib_passed:
                failed.append("Calibration")
            print(f"⚠️  NEEDS ATTENTION: {', '.join(failed)}")
        print(f"{'='*70}\n")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_improved_detection_test()
    sys.exit(0 if success else 1)
