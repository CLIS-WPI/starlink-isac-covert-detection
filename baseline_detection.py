#!/usr/bin/env python3
"""
Baseline Detection Methods
===========================
Implements simple baseline methods for comparison with CNN:
1. Power-based detection (threshold on power deviation)
2. Spectral entropy
3. Frequency-domain features + SVM
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
import json
import argparse


def load_dataset(scenario='sat'):
    """Load dataset for the given scenario (smart matching with tolerance)."""
    import glob
    import re
    
    # Try to find any matching dataset file
    scenario_letter = 'a' if scenario == 'sat' else 'b'
    pattern = f'dataset/dataset_scenario_{scenario_letter}_*.pkl'
    candidates = glob.glob(pattern)
    
    if not candidates:
        raise FileNotFoundError(f"No dataset found for scenario {scenario}")
    
    # Sort by sample count (descending) - prefer larger datasets
    def extract_samples(path):
        match = re.search(r'dataset_scenario_[ab]_(\d+)\.pkl', path)
        return int(match.group(1)) if match else 0
    
    candidates.sort(key=extract_samples, reverse=True)
    dataset_file = Path(candidates[0])
    
    print(f"  → Loading: {dataset_file.name}")
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def compute_power_deviation(rx_grids):
    """
    Compute power deviation for each sample.
    
    Args:
        rx_grids: Array of shape (N, T, F) or (N, T, H, W) or (N, T, H, W, 2)
                  Can be real or complex dtype
    
    Returns:
        Array of shape (N,) with power deviation values
    """
    # Handle complex data
    if np.iscomplexobj(rx_grids):
        # Complex data: compute power = |x|^2 = real^2 + imag^2
        power_per_element = np.abs(rx_grids) ** 2
    else:
        # Real data
        power_per_element = rx_grids ** 2
    
    # Determine the number of dimensions and sum power accordingly
    if rx_grids.ndim == 5:
        # 5D data: (N, T, H, W, 2)
        power = np.sum(power_per_element, axis=(1, 2, 3, 4))
        n_elements = rx_grids.shape[1] * rx_grids.shape[2] * rx_grids.shape[3] * rx_grids.shape[4]
    elif rx_grids.ndim == 4:
        # 4D data: (N, T, H, W)
        power = np.sum(power_per_element, axis=(1, 2, 3))
        n_elements = rx_grids.shape[1] * rx_grids.shape[2] * rx_grids.shape[3]
    elif rx_grids.ndim == 3:
        # 3D data: (N, T, F)
        power = np.sum(power_per_element, axis=(1, 2))
        n_elements = rx_grids.shape[1] * rx_grids.shape[2]
    else:
        raise ValueError(f"Unexpected number of dimensions: {rx_grids.ndim}")
    
    # Normalize by number of elements
    power = power / n_elements
    
    # Compute deviation from mean
    mean_power = np.mean(power)
    deviation = np.abs(power - mean_power) / mean_power
    
    return deviation


def compute_spectral_entropy(rx_grids, axis=1):
    """
    Compute spectral entropy for each sample.
    
    Args:
        rx_grids: Array of shape (N, T, H, W) or (N, T, H, W, 2)
        axis: Axis along which to compute FFT (default: 1 for time)
    
    Returns:
        Array of shape (N,) with spectral entropy values
    """
    N = rx_grids.shape[0]
    entropies = np.zeros(N)
    
    # Progress indicator
    print(f"  Computing spectral entropy for {N} samples...")
    progress_step = max(1, N // 20)  # Update every 5%
    
    for i in range(N):
        # Show progress
        if i % progress_step == 0 or i == N - 1:
            pct = (i + 1) / N * 100
            print(f"    Progress: {i+1}/{N} ({pct:.1f}%)", end='\r', flush=True)
        
        if rx_grids.ndim == 5:
            # Complex data: compute magnitude
            signal = rx_grids[i, :, :, :, 0] + 1j * rx_grids[i, :, :, :, 1]
        else:
            signal = rx_grids[i]
        
        # Flatten spatial dimensions and compute FFT along time axis
        signal_flat = signal.reshape(signal.shape[0], -1)  # (T, H*W)
        
        # Compute FFT and power spectrum
        fft_result = np.abs(fft(signal_flat, axis=0))
        power_spectrum = fft_result ** 2
        
        # Average over spatial dimensions
        power_spectrum = np.mean(power_spectrum, axis=1)
        
        # Normalize to get probability distribution
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Compute entropy
        entropies[i] = entropy(power_spectrum)
    
    print()  # New line after progress
    return entropies


def extract_frequency_features(rx_grids, n_features=50):
    """
    Extract frequency-domain features for SVM.
    
    Features include:
    - FFT magnitude statistics (mean, std, max, etc.)
    - Spectral entropy
    - Power in different frequency bands
    - Peak frequencies
    
    Args:
        rx_grids: Array of shape (N, T, H, W) or (N, T, H, W, 2)
        n_features: Number of frequency bins to extract
    
    Returns:
        Array of shape (N, F) with frequency features
    """
    N = rx_grids.shape[0]
    T = rx_grids.shape[1]
    
    # Initialize feature matrix
    # Features: n_features FFT bins + 5 statistics + 1 entropy = n_features + 6
    features = np.zeros((N, n_features + 6))
    
    # Progress indicator
    print(f"  Extracting frequency features for {N} samples...")
    progress_step = max(1, N // 20)  # Update every 5%
    
    for i in range(N):
        # Show progress
        if i % progress_step == 0 or i == N - 1:
            pct = (i + 1) / N * 100
            print(f"    Progress: {i+1}/{N} ({pct:.1f}%)", end='\r', flush=True)
        if rx_grids.ndim == 5:
            # Complex data
            signal = rx_grids[i, :, :, :, 0] + 1j * rx_grids[i, :, :, :, 1]
        else:
            signal = rx_grids[i]
        
        # Flatten spatial dimensions
        signal_flat = signal.reshape(signal.shape[0], -1)  # (T, H*W)
        
        # Compute FFT
        fft_result = np.abs(fft(signal_flat, axis=0))
        
        # Average over spatial dimensions
        fft_avg = np.mean(fft_result, axis=1)
        
        # Extract n_features bins (evenly spaced)
        indices = np.linspace(0, len(fft_avg)//2, n_features, dtype=int)
        features[i, :n_features] = fft_avg[indices]
        
        # Compute statistics
        features[i, n_features] = np.mean(fft_avg)
        features[i, n_features + 1] = np.std(fft_avg)
        features[i, n_features + 2] = np.max(fft_avg)
        features[i, n_features + 3] = np.min(fft_avg)
        features[i, n_features + 4] = np.median(fft_avg)
        
        # Compute spectral entropy
        power_spectrum = fft_avg ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        features[i, n_features + 5] = entropy(power_spectrum)
    
    print()  # New line after progress
    return features


def train_power_based_detector(X_train, y_train, X_test, y_test):
    """
    Power-based detection using threshold on power deviation.
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("🔋 Baseline 1: Power-Based Detection")
    print("="*80)
    
    # Compute power deviations
    train_deviations = compute_power_deviation(X_train)
    test_deviations = compute_power_deviation(X_test)
    
    # Use deviations as anomaly scores
    y_scores = test_deviations
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_scores)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute precision/recall at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ Power-Based Detection Results:")
    print(f"   • AUC: {auc:.4f}")
    print(f"   • Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1 Score: {f1:.4f}")
    
    return {
        'method': 'Power-Based Detection',
        'auc': float(auc),
        'threshold': float(optimal_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_scores': y_scores.tolist()
    }


def train_spectral_entropy_detector(X_train, y_train, X_test, y_test):
    """
    Spectral entropy-based detection.
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("📊 Baseline 2: Spectral Entropy Detection")
    print("="*80)
    
    # Compute spectral entropies
    print("Computing spectral entropies...")
    train_entropies = compute_spectral_entropy(X_train)
    test_entropies = compute_spectral_entropy(X_test)
    
    # Use entropies as anomaly scores (higher entropy = more anomalous)
    y_scores = test_entropies
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_scores)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute precision/recall at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ Spectral Entropy Detection Results:")
    print(f"   • AUC: {auc:.4f}")
    print(f"   • Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1 Score: {f1:.4f}")
    
    return {
        'method': 'Spectral Entropy',
        'auc': float(auc),
        'threshold': float(optimal_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_scores': y_scores.tolist()
    }


def train_svm_detector(X_train, y_train, X_test, y_test, n_features=50):
    """
    SVM with frequency-domain features.
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("🤖 Baseline 3: Frequency Features + SVM")
    print("="*80)
    
    # Extract frequency features
    print(f"Extracting {n_features} frequency features...")
    train_features = extract_frequency_features(X_train, n_features=n_features)
    test_features = extract_frequency_features(X_test, n_features=n_features)
    
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    # Train SVM with probability estimates
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(train_features, y_train)
    
    # Get probability scores
    y_scores = svm.predict_proba(test_features)[:, 1]
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_scores)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute precision/recall at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ SVM Detection Results:")
    print(f"   • AUC: {auc:.4f}")
    print(f"   • Optimal Threshold: {optimal_threshold:.4f}")
    print(f"   • Precision: {precision:.4f}")
    print(f"   • Recall: {recall:.4f}")
    print(f"   • F1 Score: {f1:.4f}")
    
    return {
        'method': 'Frequency Features + SVM',
        'auc': float(auc),
        'threshold': float(optimal_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_scores': y_scores.tolist(),
        'n_features': n_features
    }


def compare_with_cnn(scenario='sat', baseline_results=None):
    """
    Load CNN results and compare with baselines.
    """
    print("\n" + "="*80)
    print("📊 COMPARISON: Baselines vs CNN")
    print("="*80)
    
    # Load CNN results
    if scenario == 'sat':
        cnn_file = Path('result/scenario_a/detection_results_cnn.json')
    else:
        cnn_file = Path('result/scenario_b/detection_results_cnn.json')
    
    if cnn_file.exists():
        with open(cnn_file, 'r') as f:
            cnn_results = json.load(f)
        
        cnn_metrics = cnn_results.get('metrics', {})
        cnn_auc = cnn_metrics.get('auc', 0)
        
        # Get F1-optimal threshold metrics
        if 'dual_thresholds' in cnn_metrics:
            cnn_f1_opt = cnn_metrics['dual_thresholds'].get('f1_optimal', {})
            cnn_precision = cnn_f1_opt.get('precision', 0)
            cnn_recall = cnn_f1_opt.get('recall', 0)
            cnn_f1 = cnn_f1_opt.get('f1', 0)
        else:
            cnn_precision = cnn_metrics.get('precision', 0)
            cnn_recall = cnn_metrics.get('recall', 0)
            cnn_f1 = cnn_metrics.get('f1', 0)
        
        print(f"\n{'Method':<30} {'AUC':>10} {'Precision':>12} {'Recall':>10} {'F1 Score':>10}")
        print("-" * 80)
        
        # Print baseline results
        for result in baseline_results:
            print(f"{result['method']:<30} {result['auc']:>10.4f} {result['precision']:>12.4f} "
                  f"{result['recall']:>10.4f} {result['f1']:>10.4f}")
        
        # Print CNN results
        print("-" * 80)
        print(f"{'CNN (Our Method)':<30} {cnn_auc:>10.4f} {cnn_precision:>12.4f} "
              f"{cnn_recall:>10.4f} {cnn_f1:>10.4f}")
        print("=" * 80)
        
        # Compute improvements
        print("\n📈 CNN Improvement over Best Baseline:")
        best_baseline = max(baseline_results, key=lambda x: x['auc'])
        auc_improvement = ((cnn_auc - best_baseline['auc']) / best_baseline['auc'] * 100) if best_baseline['auc'] > 0 else 0
        f1_improvement = ((cnn_f1 - best_baseline['f1']) / best_baseline['f1'] * 100) if best_baseline['f1'] > 0 else 0
        
        print(f"   • AUC improvement: {auc_improvement:+.2f}%")
        print(f"   • F1 improvement: {f1_improvement:+.2f}%")
        print(f"   • Best baseline: {best_baseline['method']} (AUC={best_baseline['auc']:.4f})")
    else:
        print(f"⚠️  CNN results not found at {cnn_file}")
        print("   Run the CNN training first to enable comparison.")


def main():
    parser = argparse.ArgumentParser(description='Baseline Detection Methods')
    parser.add_argument('--scenario', type=str, default='sat', choices=['sat', 'ground'],
                        help='Scenario: sat (A) or ground (B)')
    parser.add_argument('--svm-features', type=int, default=50,
                        help='Number of frequency features for SVM')
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 Baseline Detection Methods Evaluation")
    print("="*80)
    print(f"Scenario: {args.scenario.upper()}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.scenario)
    
    X = dataset['rx_grids']
    y = np.array(dataset['labels'])
    
    print(f"Dataset loaded: {len(y)} samples")
    print(f"Shape: {X.shape}")
    print(f"Benign: {np.sum(y == 0)}, Attack: {np.sum(y == 1)}")
    
    # Split dataset (same as CNN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # Run baseline methods
    baseline_results = []
    
    # 1. Power-based detection
    try:
        result = train_power_based_detector(X_train, y_train, X_test, y_test)
        baseline_results.append(result)
    except Exception as e:
        print(f"❌ Power-based detection failed: {e}")
    
    # 2. Spectral entropy
    try:
        result = train_spectral_entropy_detector(X_train, y_train, X_test, y_test)
        baseline_results.append(result)
    except Exception as e:
        print(f"❌ Spectral entropy detection failed: {e}")
    
    # 3. SVM with frequency features
    try:
        result = train_svm_detector(X_train, y_train, X_test, y_test, 
                                   n_features=args.svm_features)
        baseline_results.append(result)
    except Exception as e:
        print(f"❌ SVM detection failed: {e}")
    
    # Save results
    result_dir = Path('result') / f'scenario_{"a" if args.scenario == "sat" else "b"}'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = result_dir / 'baseline_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'scenario': args.scenario,
            'baselines': baseline_results,
            'comparison_date': str(Path(output_file).stat().st_mtime) if output_file.exists() else None
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Compare with CNN
    compare_with_cnn(args.scenario, baseline_results)
    
    print("\n" + "="*80)
    print("✅ BASELINE EVALUATION COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()

