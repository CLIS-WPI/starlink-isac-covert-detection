#!/usr/bin/env python3
"""
🔧 Phase 2: Baseline Detection Methods
=====================================
Three baseline methods for comparison with CNN:
1. Energy-based detection
2. Cyclostationary analysis
3. Classical ML (feature extraction + SVM/RandomForest)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from scipy.fft import fft, fftfreq

from config.settings import GLOBAL_SEED, RESULT_DIR
from utils.reproducibility import set_global_seeds, log_seed_info

# 🔒 Phase 0: Set global seeds
log_seed_info("detector_baselines.py")
set_global_seeds(deterministic=True)


def energy_baseline(X_train, y_train, X_val, y_val, X_test, y_test, injection_subcarriers=None):
    """
    Baseline 1: Energy-based detection.
    
    Compares total energy in injection subband (or full band) between benign/attack.
    Threshold optimized on validation set.
    
    Args:
        X_train, X_val, X_test: OFDM grids (N, symbols, subcarriers) or (N, 1, 1, symbols, subcarriers)
        y_train, y_val, y_test: Labels (0=benign, 1=attack)
        injection_subcarriers: List of subcarrier indices where injection occurs (default: 24-39)
    
    Returns:
        dict: Metrics (AUC, Precision, Recall, F1) and predictions
    """
    print("\n" + "="*70)
    print("🔍 Baseline 1: Energy-Based Detection")
    print("="*70)
    
    # 🔧 FIX: Reshape if needed - handle 5D shapes properly
    if X_train.ndim == 5:
        # Squeeze batch and channel dimensions: (N, 1, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=(1, 2))
        X_val = np.squeeze(X_val, axis=(1, 2))
        X_test = np.squeeze(X_test, axis=(1, 2))
    elif X_train.ndim == 4:
        # Handle 4D: (N, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=1)
        X_val = np.squeeze(X_val, axis=1)
        X_test = np.squeeze(X_test, axis=1)
    
    # Default injection subcarriers (middle band: 24-39)
    if injection_subcarriers is None:
        injection_subcarriers = list(range(24, 40))
    
    # Compute energy in injection subband for each sample
    def compute_energy(X, subcarriers):
        """Compute total energy in specified subcarriers."""
        energies = []
        for i in range(len(X)):
            grid = X[i]  # Should be (symbols, subcarriers) after squeeze
            # 🔧 FIX: Check grid shape and handle edge cases
            if grid.ndim > 2:
                # Try to squeeze again if still multi-dimensional
                grid = np.squeeze(grid)
            if grid.ndim != 2:
                print(f"  ⚠️ Warning: Unexpected grid shape {grid.shape}, skipping sample {i}")
                energies.append(0.0)
                continue
            num_subcarriers = grid.shape[1]
            # Filter subcarriers to valid range
            valid_subcarriers = [sc for sc in subcarriers if 0 <= sc < num_subcarriers]
            if len(valid_subcarriers) == 0:
                print(f"  ⚠️ Warning: No valid subcarriers for sample {i} (grid has {num_subcarriers} subcarriers)")
                energies.append(0.0)
                continue
            # Extract energy in injection subcarriers
            energy = np.sum(np.abs(grid[:, valid_subcarriers])**2)
            energies.append(energy)
        return np.array(energies)
    
    train_energies = compute_energy(X_train, injection_subcarriers)
    val_energies = compute_energy(X_val, injection_subcarriers)
    test_energies = compute_energy(X_test, injection_subcarriers)
    
    print(f"  ✓ Injection subcarriers: {injection_subcarriers}")
    print(f"  ✓ Train energy - Benign: {np.mean(train_energies[y_train==0]):.6e}, "
          f"Attack: {np.mean(train_energies[y_train==1]):.6e}")
    
    # Optimize threshold on validation set (maximize F1)
    best_threshold = None
    best_f1 = 0.0
    
    # Search threshold range
    min_energy = np.min(val_energies)
    max_energy = np.max(val_energies)
    thresholds = np.linspace(min_energy, max_energy, 1000)
    
    for threshold in thresholds:
        y_pred = (val_energies >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"  ✓ Optimal threshold: {best_threshold:.6e} (F1={best_f1:.4f} on val)")
    
    # Evaluate on test set
    y_pred_test = (test_energies >= best_threshold).astype(int)
    y_proba_test = (test_energies - min_energy) / (max_energy - min_energy + 1e-12)
    y_proba_test = np.clip(y_proba_test, 0, 1)
    
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print(f"  ✓ Test Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f}")
    
    return {
        'method': 'Energy',
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(best_threshold),
        'y_pred': y_pred_test,
        'y_proba': y_proba_test
    }


def cyclostationary_baseline(X_train, y_train, X_val, y_val, X_test, y_test, 
                             alphas=None, downsample_factor=1):
    """
    Baseline 2: Cyclostationary analysis.
    
    Estimates cyclic autocorrelation/SCF at selected cyclic frequencies α.
    Uses summary statistic (e.g., peak in subband).
    
    Args:
        X_train, X_val, X_test: OFDM grids
        y_train, y_val, y_test: Labels
        alphas: List of cyclic frequencies to analyze (default: [0, 1/T, 2/T])
        downsample_factor: Downsampling factor for speed (default: 1 = no downsampling)
    
    Returns:
        dict: Metrics and predictions
    """
    print("\n" + "="*70)
    print("🔍 Baseline 2: Cyclostationary Analysis")
    print("="*70)
    
    # 🔧 FIX: Reshape if needed - handle 5D shapes properly
    if X_train.ndim == 5:
        # Squeeze batch and channel dimensions: (N, 1, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=(1, 2))
        X_val = np.squeeze(X_val, axis=(1, 2))
        X_test = np.squeeze(X_test, axis=(1, 2))
    elif X_train.ndim == 4:
        # Handle 4D: (N, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=1)
        X_val = np.squeeze(X_val, axis=1)
        X_test = np.squeeze(X_test, axis=1)
    
    # Default cyclic frequencies (α = 0, 1/T, 2/T where T is symbol period)
    if alphas is None:
        num_symbols = X_train.shape[1]
        alphas = [0.0, 1.0/num_symbols, 2.0/num_symbols]
    
    print(f"  ✓ Cyclic frequencies (α): {alphas}")
    print(f"  ✓ Downsample factor: {downsample_factor}")
    
    def compute_cyclic_features(X, alphas, downsample):
        """Compute cyclic autocorrelation features."""
        features = []
        for i in range(len(X)):
            grid = X[i]  # (symbols, subcarriers)
            
            # Downsample if needed
            if downsample > 1:
                grid = grid[::downsample, :]
            
            # Convert to time domain (simplified: use IFFT per symbol)
            time_domain = []
            for sym_idx in range(grid.shape[0]):
                sym_freq = grid[sym_idx, :]
                sym_time = np.fft.ifft(sym_freq)
                time_domain.append(sym_time)
            time_domain = np.array(time_domain).flatten()  # Flatten to 1D
            
            # Compute cyclic autocorrelation for each α
            alpha_features = []
            for alpha in alphas:
                # Simplified cyclic autocorrelation: R_x^α(τ) = E[x(t)x*(t-τ)e^{-j2παt}]
                # For speed, use a simplified version: correlation at specific lags
                max_lag = min(100, len(time_domain) // 4)
                lags = np.arange(1, max_lag)
                
                # Compute cyclic autocorrelation at lag τ=0 (simplified)
                if alpha == 0:
                    # Regular autocorrelation
                    autocorr = np.abs(np.correlate(time_domain, time_domain, mode='valid'))
                    feature = np.max(autocorr) if len(autocorr) > 0 else 0.0
                else:
                    # Cyclic autocorrelation (simplified)
                    # Use FFT-based approach for speed
                    fft_signal = fft(time_domain)
                    # Cyclic periodogram approximation
                    cyclic_power = np.abs(fft_signal)**2
                    # Extract peak in frequency domain
                    feature = np.max(cyclic_power) if len(cyclic_power) > 0 else 0.0
                
                alpha_features.append(feature)
            
            features.append(alpha_features)
        
        return np.array(features)
    
    print("  ✓ Computing cyclic features...")
    train_features = compute_cyclic_features(X_train, alphas, downsample_factor)
    val_features = compute_cyclic_features(X_val, alphas, downsample_factor)
    test_features = compute_cyclic_features(X_test, alphas, downsample_factor)
    
    print(f"  ✓ Feature shape: {train_features.shape}")
    
    # Use maximum feature as detector (simple threshold)
    train_max = np.max(train_features, axis=1)
    val_max = np.max(val_features, axis=1)
    test_max = np.max(test_features, axis=1)
    
    # Optimize threshold on validation set
    best_threshold = None
    best_f1 = 0.0
    
    min_val = np.min(val_max)
    max_val = np.max(val_max)
    thresholds = np.linspace(min_val, max_val, 1000)
    
    for threshold in thresholds:
        y_pred = (val_max >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"  ✓ Optimal threshold: {best_threshold:.6e} (F1={best_f1:.4f} on val)")
    
    # Evaluate on test set
    y_pred_test = (test_max >= best_threshold).astype(int)
    y_proba_test = (test_max - min_val) / (max_val - min_val + 1e-12)
    y_proba_test = np.clip(y_proba_test, 0, 1)
    
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print(f"  ✓ Test Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f}")
    
    return {
        'method': 'Cyclostationary',
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(best_threshold),
        'y_pred': y_pred_test,
        'y_proba': y_proba_test
    }


def classical_ml_baseline(X_train, y_train, X_val, y_val, X_test, y_test, 
                         classifier='svm'):
    """
    Baseline 3: Classical ML with feature extraction.
    
    Extracts simple features from |FFT| and spectral differences between benign/attack.
    Uses SVM or RandomForest.
    
    Args:
        X_train, X_val, X_test: OFDM grids
        y_train, y_val, y_test: Labels
        classifier: 'svm' or 'rf' (RandomForest)
    
    Returns:
        dict: Metrics and predictions
    """
    print("\n" + "="*70)
    print(f"🔍 Baseline 3: Classical ML ({classifier.upper()})")
    print("="*70)
    
    # 🔧 FIX: Reshape if needed - handle 5D shapes properly
    if X_train.ndim == 5:
        # Squeeze batch and channel dimensions: (N, 1, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=(1, 2))
        X_val = np.squeeze(X_val, axis=(1, 2))
        X_test = np.squeeze(X_test, axis=(1, 2))
    elif X_train.ndim == 4:
        # Handle 4D: (N, 1, symbols, subcarriers) -> (N, symbols, subcarriers)
        X_train = np.squeeze(X_train, axis=1)
        X_val = np.squeeze(X_val, axis=1)
        X_test = np.squeeze(X_test, axis=1)
    
    def extract_features(X):
        """Extract features from OFDM grids."""
        features = []
        for i in range(len(X)):
            grid = X[i]  # Should be (symbols, subcarriers) after squeeze
            
            # 🔧 FIX: Ensure grid is 2D - squeeze again if needed
            if grid.ndim > 2:
                grid = np.squeeze(grid)
            if grid.ndim != 2:
                print(f"  ⚠️ Warning: Unexpected grid shape {grid.shape} at sample {i}, skipping")
                # Create dummy features with correct size
                n_subs = 64  # Default
                dummy_features = np.concatenate([
                    np.zeros(n_subs),  # mean_mag_per_sub
                    np.zeros(n_subs),  # var_per_sub
                    np.zeros(n_subs),  # spectral_diff
                    [0.0, 0.0, 0.0],  # energy bands
                    [0.0, 0.0, 0.0, 0.0]  # moments
                ])
                features.append(dummy_features)
                continue
            
            # Feature 1: Mean magnitude per subcarrier (averaged over symbols)
            mean_mag_per_sub = np.mean(np.abs(grid), axis=0)
            # 🔧 FIX: Ensure 1D array
            mean_mag_per_sub = np.atleast_1d(mean_mag_per_sub).flatten()
            
            # Feature 2: Variance per subcarrier
            var_per_sub = np.var(np.abs(grid), axis=0)
            # 🔧 FIX: Ensure 1D array
            var_per_sub = np.atleast_1d(var_per_sub).flatten()
            
            # Feature 3: Spectral difference (difference between adjacent subcarriers)
            spectral_diff = np.diff(mean_mag_per_sub)
            # 🔧 FIX: Pad spectral_diff to match size of mean_mag_per_sub (64 vs 63)
            # Use the last value to pad (maintains continuity)
            if len(spectral_diff) < len(mean_mag_per_sub):
                spectral_diff = np.pad(spectral_diff, (0, len(mean_mag_per_sub) - len(spectral_diff)), 
                                       mode='edge')  # Pad with edge value
            # 🔧 FIX: Ensure 1D array
            spectral_diff = np.atleast_1d(spectral_diff).flatten()
            
            # Feature 4: Energy in different bands
            n_subs = grid.shape[1]
            energy_low = np.sum(np.abs(grid[:, :n_subs//3])**2)
            energy_mid = np.sum(np.abs(grid[:, n_subs//3:2*n_subs//3])**2)
            energy_high = np.sum(np.abs(grid[:, 2*n_subs//3:])**2)
            
            # Feature 5: Statistical moments of magnitude
            mag_flat = np.abs(grid).flatten()
            mean_mag = np.mean(mag_flat)
            std_mag = np.std(mag_flat)
            skew_mag = np.mean(((mag_flat - mean_mag) / (std_mag + 1e-12))**3)
            kurt_mag = np.mean(((mag_flat - mean_mag) / (std_mag + 1e-12))**4)
            
            # 🔧 FIX: Ensure all feature arrays have consistent sizes
            n_features_per_type = len(mean_mag_per_sub)  # Should be 64
            assert len(mean_mag_per_sub) == n_features_per_type, f"Mismatch: mean_mag_per_sub has {len(mean_mag_per_sub)} elements"
            assert len(var_per_sub) == n_features_per_type, f"Mismatch: var_per_sub has {len(var_per_sub)} elements"
            assert len(spectral_diff) == n_features_per_type, f"Mismatch: spectral_diff has {len(spectral_diff)} elements (expected {n_features_per_type})"
            
            # Combine features
            feature_vector = np.concatenate([
                mean_mag_per_sub,           # n_features_per_type features
                var_per_sub,                # n_features_per_type features
                spectral_diff,              # n_features_per_type features (padded)
                [energy_low, energy_mid, energy_high],  # 3 features
                [mean_mag, std_mag, skew_mag, kurt_mag]  # 4 features
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    print("  ✓ Extracting features...")
    train_features = extract_features(X_train)
    val_features = extract_features(X_val)
    test_features = extract_features(X_test)
    
    print(f"  ✓ Feature dimension: {train_features.shape[1]}")
    
    # Train classifier
    if classifier.lower() == 'svm':
        clf = SVC(kernel='rbf', probability=True, random_state=GLOBAL_SEED)
    elif classifier.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    print(f"  ✓ Training {classifier.upper()}...")
    clf.fit(train_features, y_train)
    
    # Evaluate on test set
    y_pred_test = clf.predict(test_features)
    y_proba_test = clf.predict_proba(test_features)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print(f"  ✓ Test Results:")
    print(f"      AUC: {auc:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall: {recall:.4f}")
    print(f"      F1: {f1:.4f}")
    
    return {
        'method': f'Classical-ML-{classifier.upper()}',
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_pred': y_pred_test,
        'y_proba': y_proba_test
    }


def main():
    """Run all baseline detectors and compare with CNN."""
    parser = argparse.ArgumentParser(description="Phase 2: Baseline detection methods")
    parser.add_argument('--dataset', type=str, default=None, help="Path to dataset file (auto-detected if not provided)")
    parser.add_argument('--output-csv', type=str, default=None, help="Output CSV path")
    parser.add_argument('--skip-cyclo', action='store_true', help="Skip cyclostationary (slow)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 PHASE 2: BASELINE DETECTION METHODS")
    print("="*70)
    
    # 🔧 FIX: Auto-detect latest dataset if not provided or if provided path doesn't exist
    if args.dataset is None or not os.path.exists(args.dataset):
        import glob
        from config.settings import INSIDER_MODE
        
        # Determine scenario name from dataset path or INSIDER_MODE
        if args.dataset:
            scenario_name = 'scenario_a' if 'scenario_a' in args.dataset else 'scenario_b' if 'scenario_b' in args.dataset else ('scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b')
        else:
            scenario_name = 'scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b'
        
        # Find latest scenario-specific dataset
        dataset_files = glob.glob(os.path.join("dataset", f"dataset_{scenario_name}*.pkl"))
        if dataset_files:
            # Sort by modification time (newest first)
            dataset_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            dataset_path = dataset_files[0]  # Latest/newest
            if args.dataset is None:
                print(f"  → Auto-detected latest dataset: {os.path.basename(dataset_path)}")
            else:
                print(f"  ⚠️  Provided dataset not found, using latest: {os.path.basename(dataset_path)}")
        else:
            dataset_path = args.dataset or f"dataset/dataset_{scenario_name}.pkl"
            if args.dataset is None:
                print(f"  → Using default: {dataset_path}")
    else:
        dataset_path = args.dataset
    
    print(f"Dataset: {dataset_path}")
    print("="*70)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        return False
    
    print(f"\n📂 Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract data
    X_grids = dataset['tx_grids']  # Use pre-channel grids for stronger pattern
    Y = dataset['labels']
    
    print(f"✓ Dataset loaded: {len(Y)} samples")
    print(f"✓ Benign: {np.sum(Y==0)}, Attack: {np.sum(Y==1)}")
    
    # Split data (70/20/10: train/val/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_grids, Y,
        test_size=0.3,
        random_state=GLOBAL_SEED,
        stratify=Y
    )
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2/0.7,  # 20% of total
        random_state=GLOBAL_SEED,
        stratify=y_train
    )
    
    print(f"✓ Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Run baseline detectors
    results = []
    
    # Baseline 1: Energy
    try:
        result1 = energy_baseline(X_tr, y_tr, X_val, y_val, X_test, y_test)
        results.append(result1)
    except Exception as e:
        print(f"  ❌ Energy baseline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Baseline 2: Cyclostationary
    if not args.skip_cyclo:
        try:
            result2 = cyclostationary_baseline(X_tr, y_tr, X_val, y_val, X_test, y_test,
                                               downsample_factor=2)  # Downsample for speed
            results.append(result2)
        except Exception as e:
            print(f"  ❌ Cyclostationary baseline failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Baseline 3: Classical ML (SVM)
    try:
        result3 = classical_ml_baseline(X_tr, y_tr, X_val, y_val, X_test, y_test, classifier='svm')
        results.append(result3)
    except Exception as e:
        print(f"  ❌ Classical ML (SVM) baseline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Baseline 3b: Classical ML (RandomForest)
    try:
        result4 = classical_ml_baseline(X_tr, y_tr, X_val, y_val, X_test, y_test, classifier='rf')
        results.append(result4)
    except Exception as e:
        print(f"  ❌ Classical ML (RF) baseline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📊 BASELINE RESULTS SUMMARY")
    print("="*70)
    
    summary_data = []
    for r in results:
        print(f"{r['method']:20s} | AUC: {r['auc']:.4f} | Precision: {r['precision']:.4f} | "
              f"Recall: {r['recall']:.4f} | F1: {r['f1']:.4f}")
        summary_data.append({
            'method': r['method'],
            'auc': r['auc'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1']
        })
    
    # Save to CSV
    scenario = 'scenario_a' if 'scenario_a' in dataset_path else 'scenario_b' if 'scenario_b' in dataset_path else 'unknown'
    output_csv = args.output_csv or f"{RESULT_DIR}/baselines_{scenario}.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

