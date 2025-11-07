#!/usr/bin/env python3
"""
🔧 Phase 3: Robustness Analysis
================================
Evaluate CNN model robustness across different parameter configurations:
- SNR sweep
- Amplitude sweep
- Pattern variation (fixed vs random)
- Subband variation (mid vs random16)
- Doppler scale sweep

Generates CSV results and visualization plots.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from config.settings import GLOBAL_SEED, MODEL_DIR, RESULT_DIR, DATASET_DIR
from utils.reproducibility import set_global_seeds, log_seed_info
from model.detector_cnn import CNNDetector

# 🔒 Phase 0: Set global seeds
log_seed_info("sweep_eval.py")
set_global_seeds(deterministic=True)


def load_trained_model(scenario, use_csi=False):
    """
    Load trained CNN model for a scenario.
    
    Args:
        scenario: 'sat' (scenario_a) or 'ground' (scenario_b)
        use_csi: Whether to load CNN+CSI model (default: CNN-only)
    
    Returns:
        CNNDetector: Loaded model
    """
    scenario_folder = 'scenario_a' if scenario == 'sat' else 'scenario_b'
    model_suffix = '_csi' if use_csi else ''
    model_filename = f"cnn_detector{model_suffix}.keras"
    model_path = f"{MODEL_DIR}/{scenario_folder}/{model_filename}"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\n"
                               f"Please train the model first:\n"
                               f"  python3 main_detection_cnn.py --scenario {scenario} --epochs 50")
    
    print(f"  ✓ Loading model from: {model_path}")
    
    # Load model
    detector = CNNDetector(use_csi=use_csi)
    detector.load(model_path)
    
    return detector


def filter_dataset_by_params(dataset, snr_list=None, amp_list=None, 
                             pattern_list=None, subband_list=None, 
                             doppler_scale_list=None):
    """
    Filter dataset by parameter values.
    
    Args:
        dataset: Dataset dictionary with 'meta' key
        snr_list: List of SNR values to include (None = all)
        amp_list: List of amplitude values to include (None = all)
        pattern_list: List of patterns to include (None = all)
        subband_list: List of subbands to include (None = all)
        doppler_scale_list: List of Doppler scales to include (None = all)
    
    Returns:
        dict: Filtered dataset with indices
    """
    if 'meta' not in dataset or not dataset['meta']:
        print("  ⚠️  Warning: No metadata found, returning all samples")
        return {
            'indices': np.arange(len(dataset['labels'])),
            'filtered': False
        }
    
    indices = []
    for i, meta in enumerate(dataset['meta']):
        if not isinstance(meta, dict):
            continue
        
        # Check filters
        include = True
        
        if snr_list is not None and meta.get('snr_db') not in snr_list:
            include = False
        
        if amp_list is not None and meta.get('covert_amp') not in amp_list:
            include = False
        
        if pattern_list is not None:
            pattern = meta.get('pattern') or meta.get('injection_info', {}).get('pattern')
            if pattern not in pattern_list:
                include = False
        
        if subband_list is not None:
            subband = meta.get('subband_mode') or meta.get('injection_info', {}).get('subband_mode')
            if subband not in subband_list:
                include = False
        
        if doppler_scale_list is not None and meta.get('doppler_scale') not in doppler_scale_list:
            include = False
        
        if include:
            indices.append(i)

    indices = np.array(indices, dtype=int)

    class_counts = None
    balanced = True
    if len(indices) > 0:
        labels_subset = np.asarray(dataset['labels'])[indices]
        unique_labels, counts = np.unique(labels_subset, return_counts=True)
        class_counts = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
        balanced = len(unique_labels) >= 2

    return {
        'indices': indices,
        'filtered': len(indices) < len(dataset['labels']),
        'balanced': balanced,
        'class_counts': class_counts
    }


def evaluate_on_subset(detector, dataset, indices, use_csi=False):
    """
    Evaluate detector on a subset of the dataset.
    
    Args:
        detector: CNNDetector instance
        dataset: Dataset dictionary
        indices: Indices of samples to evaluate
        use_csi: Whether to use CSI
    
    Returns:
        dict: Metrics (auc, precision, recall, f1) and predictions
    """
    if len(indices) == 0:
        return {
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'n_samples': 0
        }

    # Extract subset - ✅ REALISTIC: Use rx_grids (post-channel) for evaluation
    # This matches real-world scenario where detector only sees post-channel signals
    if 'rx_grids' in dataset:
        X_grids = dataset['rx_grids'][indices]
    elif 'tx_grids' in dataset:
        X_grids = dataset['tx_grids'][indices]
        print(f"  ⚠️  Using tx_grids (rx_grids not available, fallback)")
    else:
        raise ValueError("Neither rx_grids nor tx_grids found in dataset")
    Y = dataset['labels'][indices]

    unique_labels = np.unique(Y)
    if len(unique_labels) < 2:
        counts = {int(lbl): int(np.sum(Y == lbl)) for lbl in unique_labels}
        print(f"  ⚠️  Skipping evaluation: subset contains a single class (counts={counts})")
        return {
            'auc': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'n_samples': len(indices),
            'class_counts': counts
        }

    if use_csi and 'csi_est' in dataset and dataset['csi_est'] is not None:
        X_csi = dataset['csi_est'][indices]
    else:
        X_csi = None
    
    # Evaluate with threshold optimization
    try:
        # Get predictions and probabilities first
        y_proba = detector.predict_proba(X_grids, X_csi_test=X_csi)
        
        # ✅ THRESHOLD OPTIMIZATION: Find optimal threshold for F1-max
        # Split into threshold optimization set (40%) and evaluation set (60%)
        from sklearn.model_selection import train_test_split
        if len(Y) > 20:  # Only split if enough samples
            try:
                # Split indices first to ensure consistency
                indices = np.arange(len(Y))
                idx_thresh, idx_eval = train_test_split(
                    indices,
                    test_size=0.6,
                    stratify=Y,
                    random_state=42
                )
                
                # Split data using consistent indices
                X_thresh = X_grids[idx_thresh]
                X_eval = X_grids[idx_eval]
                y_thresh = Y[idx_thresh]
                y_eval = Y[idx_eval]
                
                if X_csi is not None:
                    X_csi_thresh = X_csi[idx_thresh]
                    X_csi_eval = X_csi[idx_eval]
                else:
                    X_csi_thresh = None
                    X_csi_eval = None
                
                # Optimize threshold on threshold set
                optimal_threshold = detector.find_optimal_threshold(
                    X_thresh, y_thresh,
                    X_csi_val=X_csi_thresh,
                    metric='f1'  # Maximize F1 score
                )
                
                # Evaluate on eval set with optimal threshold
                y_proba_eval = detector.predict_proba(X_eval, X_csi_test=X_csi_eval)
                y_pred_eval = (y_proba_eval >= optimal_threshold).astype(int)
                
                # Calculate metrics
                from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
                auc = roc_auc_score(y_eval, y_proba_eval) if len(np.unique(y_eval)) > 1 else 0.0
                precision = precision_score(y_eval, y_pred_eval, zero_division=0)
                recall = recall_score(y_eval, y_pred_eval, zero_division=0)
                f1 = f1_score(y_eval, y_pred_eval, zero_division=0)
                threshold = optimal_threshold
                
                # Use eval set for final results
                Y_final = y_eval
                y_proba_final = y_proba_eval
            except Exception as e:
                print(f"  ⚠️  Threshold optimization failed: {e}, using default evaluation")
                # Fallback to default evaluation
                results = detector.evaluate(X_grids, Y, X_csi_test=X_csi)
                auc = results.get('auc', 0.0)
                precision = results.get('precision', 0.0)
                recall = results.get('recall', 0.0)
                f1 = results.get('f1', 0.0)
                threshold = results.get('threshold', 0.5)
                Y_final = Y
                y_proba_final = y_proba
        else:
            # Too few samples, use default evaluation
            results = detector.evaluate(X_grids, Y, X_csi_test=X_csi)
            auc = results.get('auc', 0.0)
            precision = results.get('precision', 0.0)
            recall = results.get('recall', 0.0)
            f1 = results.get('f1', 0.0)
            threshold = results.get('threshold', 0.5)
            Y_final = Y
            y_proba_final = y_proba
        
        return {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(threshold),
            'n_samples': len(indices),
            'y_proba': y_proba_final if 'y_proba_final' in locals() else y_proba,
            'y_true': Y_final if 'Y_final' in locals() else Y,
            'class_counts': {int(lbl): int(np.sum((Y_final if 'Y_final' in locals() else Y) == lbl)) for lbl in unique_labels}
        }
    except Exception as e:
        print(f"  ⚠️  Evaluation error: {e}")
        return {
            'auc': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'n_samples': len(indices)
        }


def sweep_snr(detector, dataset, snr_list, use_csi=False):
    """Sweep SNR values and evaluate."""
    print(f"\n📊 Sweeping SNR: {snr_list}")
    results = []
    
    for snr in snr_list:
        filter_result = filter_dataset_by_params(dataset, snr_list=[snr])
        indices = filter_result['indices']
        
        if len(indices) == 0:
            print(f"  ⚠️  No samples found for SNR={snr} dB")
            continue

        if not filter_result.get('balanced', True):
            print(f"  ⚠️  Skipping SNR={snr} dB due to single-class subset (counts={filter_result.get('class_counts')})")
            continue

        print(f"  SNR={snr} dB: {len(indices)} samples")
        eval_result = evaluate_on_subset(detector, dataset, indices, use_csi=use_csi)
        eval_result['snr_db'] = snr
        results.append(eval_result)
    
    return results


def sweep_amplitude(detector, dataset, amp_list, use_csi=False):
    """Sweep amplitude values and evaluate."""
    print(f"\n📊 Sweeping Amplitude: {amp_list}")
    results = []
    
    for amp in amp_list:
        filter_result = filter_dataset_by_params(dataset, amp_list=[amp])
        indices = filter_result['indices']
        
        if len(indices) == 0:
            print(f"  ⚠️  No samples found for amp={amp}")
            continue

        if not filter_result.get('balanced', True):
            print(f"  ⚠️  Skipping amp={amp} due to single-class subset (counts={filter_result.get('class_counts')})")
            continue

        print(f"  Amp={amp}: {len(indices)} samples")
        eval_result = evaluate_on_subset(detector, dataset, indices, use_csi=use_csi)
        eval_result['covert_amp'] = amp
        results.append(eval_result)
    
    return results


def sweep_pattern(detector, dataset, pattern_list, use_csi=False):
    """Sweep pattern types and evaluate."""
    print(f"\n📊 Sweeping Pattern: {pattern_list}")
    results = []
    
    for pattern in pattern_list:
        filter_result = filter_dataset_by_params(dataset, pattern_list=[pattern])
        indices = filter_result['indices']
        
        if len(indices) == 0:
            print(f"  ⚠️  No samples found for pattern={pattern}")
            continue

        if not filter_result.get('balanced', True):
            print(f"  ⚠️  Skipping pattern={pattern} due to single-class subset (counts={filter_result.get('class_counts')})")
            continue

        print(f"  Pattern={pattern}: {len(indices)} samples")
        eval_result = evaluate_on_subset(detector, dataset, indices, use_csi=use_csi)
        eval_result['pattern'] = pattern
        results.append(eval_result)
    
    return results


def sweep_doppler_scale(detector, dataset, doppler_scale_list, use_csi=False):
    """Sweep Doppler scale values and evaluate."""
    print(f"\n📊 Sweeping Doppler Scale: {doppler_scale_list}")
    results = []
    
    for scale in doppler_scale_list:
        filter_result = filter_dataset_by_params(dataset, doppler_scale_list=[scale])
        indices = filter_result['indices']
        
        if len(indices) == 0:
            print(f"  ⚠️  No samples found for doppler_scale={scale}")
            continue

        if not filter_result.get('balanced', True):
            print(f"  ⚠️  Skipping doppler_scale={scale} due to single-class subset (counts={filter_result.get('class_counts')})")
            continue

        print(f"  Doppler Scale={scale}: {len(indices)} samples")
        eval_result = evaluate_on_subset(detector, dataset, indices, use_csi=use_csi)
        eval_result['doppler_scale'] = scale
        results.append(eval_result)
    
    return results


def plot_robustness_results(results_dict, output_dir, scenario):
    """Generate robustness plots."""
    print(f"\n📈 Generating robustness plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: AUC vs SNR
    if 'snr' in results_dict and len(results_dict['snr']) > 0:
        snr_results = results_dict['snr']
        snr_pairs = [(r['snr_db'], r['auc']) for r in snr_results if r.get('auc') is not None]
        if snr_pairs:
            snr_values, auc_values = zip(*snr_pairs)
            plt.figure(figsize=(8, 6))
            plt.plot(snr_values, auc_values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('SNR (dB)', fontsize=12)
            plt.ylabel('AUC', fontsize=12)
            plt.title(f'Robustness: AUC vs SNR ({scenario})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_auc_vs_snr_{scenario}.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: robustness_auc_vs_snr_{scenario}.png")

    # Plot 2: AUC vs Amplitude
    if 'amplitude' in results_dict and len(results_dict['amplitude']) > 0:
        amp_results = results_dict['amplitude']
        amp_pairs = [(r['covert_amp'], r['auc']) for r in amp_results if r.get('auc') is not None]
        if amp_pairs:
            amp_values, auc_values = zip(*amp_pairs)
            plt.figure(figsize=(8, 6))
            plt.plot(amp_values, auc_values, 's-', linewidth=2, markersize=8, color='green')
            plt.xlabel('Covert Amplitude', fontsize=12)
            plt.ylabel('AUC', fontsize=12)
            plt.title(f'Robustness: AUC vs Amplitude ({scenario})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_auc_vs_amp_{scenario}.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: robustness_auc_vs_amp_{scenario}.png")

    # Plot 3: Boxplot AUC for fixed vs random pattern
    if 'pattern' in results_dict and len(results_dict['pattern']) > 0:
        pattern_results = results_dict['pattern']
        pattern_data = {}
        for r in pattern_results:
            pattern = r.get('pattern', 'unknown')
            auc_value = r.get('auc')
            if auc_value is None:
                continue
            if pattern not in pattern_data:
                pattern_data[pattern] = []
            pattern_data[pattern].append(auc_value)
        
        if len(pattern_data) > 0:
            plt.figure(figsize=(8, 6))
            patterns = list(pattern_data.keys())
            auc_lists = [pattern_data[p] for p in patterns]
            
            bp = plt.boxplot(auc_lists, tick_labels=patterns, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            plt.ylabel('AUC', fontsize=12)
            plt.xlabel('Pattern Type', fontsize=12)
            plt.title(f'Robustness: AUC Distribution by Pattern ({scenario})', fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_auc_by_pattern_{scenario}.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: robustness_auc_by_pattern_{scenario}.png")
    
    # Plot 4: AUC vs Doppler Scale
    if 'doppler' in results_dict and len(results_dict['doppler']) > 0:
        doppler_results = results_dict['doppler']
        doppler_pairs = [(r['doppler_scale'], r['auc']) for r in doppler_results if r.get('auc') is not None]
        if doppler_pairs:
            scale_values, auc_values = zip(*doppler_pairs)
            plt.figure(figsize=(8, 6))
            plt.plot(scale_values, auc_values, '^-', linewidth=2, markersize=8, color='orange')
            plt.xlabel('Doppler Scale Factor', fontsize=12)
            plt.ylabel('AUC', fontsize=12)
            plt.title(f'Robustness: AUC vs Doppler Scale ({scenario})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_auc_vs_doppler_{scenario}.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved: robustness_auc_vs_doppler_{scenario}.png")


def main():
    """Main robustness evaluation."""
    parser = argparse.ArgumentParser(description="Phase 3: Robustness analysis")
    parser.add_argument('--scenario', type=str, choices=['sat', 'ground'], required=True,
                       help="Scenario: 'sat' (scenario_a) or 'ground' (scenario_b)")
    parser.add_argument('--dataset', type=str, default=None,
                       help="Path to dataset file (auto-detect if not provided)")
    parser.add_argument('--use-csi', action='store_true',
                       help="Use CNN+CSI model (default: CNN-only)")
    
    # Sweep parameters
    parser.add_argument('--snr-list', type=str, default=None,
                       help="Comma-separated SNR values (e.g., '-5,0,5,10,15,20')")
    parser.add_argument('--amp-list', type=str, default=None,
                       help="Comma-separated amplitude values (e.g., '0.1,0.3,0.5,0.7')")
    parser.add_argument('--pattern', type=str, default=None,
                       help="Comma-separated patterns (e.g., 'fixed,random')")
    parser.add_argument('--subband', type=str, default=None,
                       help="Comma-separated subbands (e.g., 'mid,random16')")
    parser.add_argument('--doppler-scale-list', type=str, default=None,
                       help="Comma-separated Doppler scales (e.g., '0.5,1.0,1.5')")
    
    # Output
    parser.add_argument('--output-csv', type=str, default=None,
                       help="Output CSV path (default: result/robustness_{scenario}.csv)")
    parser.add_argument('--output-dir', type=str, default=None,
                       help="Output directory for plots (default: result/)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 PHASE 3: ROBUSTNESS ANALYSIS")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Model: {'CNN+CSI' if args.use_csi else 'CNN-only'}")
    print("="*70)
    
    # Auto-detect dataset
    if args.dataset is None:
        scenario_name = 'scenario_a' if args.scenario == 'sat' else 'scenario_b'
        dataset_files = [
            f"{DATASET_DIR}/dataset_{scenario_name}_10k.pkl",
            f"{DATASET_DIR}/dataset_{scenario_name}.pkl"
        ]
        
        for df in dataset_files:
            if os.path.exists(df):
                args.dataset = df
                break
        
        if args.dataset is None:
            print(f"❌ Dataset not found. Please provide --dataset or generate Phase 1 dataset.")
            return False
    
    # Load dataset
    print(f"\n📂 Loading dataset: {args.dataset}")
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Dataset loaded: {len(dataset['labels'])} samples")
    
    # Load trained model
    print(f"\n🤖 Loading trained model...")
    try:
        detector = load_trained_model(args.scenario, use_csi=args.use_csi)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    # Parse sweep parameters
    snr_list = None
    if args.snr_list:
        snr_list = [float(x.strip()) for x in args.snr_list.split(',')]
    
    amp_list = None
    if args.amp_list:
        amp_list = [float(x.strip()) for x in args.amp_list.split(',')]
        # ✅ FILTER: Remove COVERT_AMP < 0.3 from main results (too weak for detection)
        amp_list = [a for a in amp_list if a >= 0.3]
        if len(amp_list) < len([float(x.strip()) for x in args.amp_list.split(',')]):
            print(f"  ℹ️  Filtered out COVERT_AMP < 0.3 (too weak for detection)")
            print(f"  ℹ️  Remaining amplitudes: {amp_list}")
    
    pattern_list = None
    if args.pattern:
        pattern_list = [x.strip() for x in args.pattern.split(',')]
    
    subband_list = None
    if args.subband:
        subband_list = [x.strip() for x in args.subband.split(',')]
    
    doppler_scale_list = None
    if args.doppler_scale_list:
        doppler_scale_list = [float(x.strip()) for x in args.doppler_scale_list.split(',')]
    
    # Run sweeps
    results_dict = {}
    all_results = []
    
    if snr_list:
        snr_results = sweep_snr(detector, dataset, snr_list, use_csi=args.use_csi)
        results_dict['snr'] = snr_results
        for r in snr_results:
            all_results.append({**r, 'sweep_type': 'snr'})
    
    if amp_list:
        amp_results = sweep_amplitude(detector, dataset, amp_list, use_csi=args.use_csi)
        results_dict['amplitude'] = amp_results
        for r in amp_results:
            all_results.append({**r, 'sweep_type': 'amplitude'})
    
    if pattern_list:
        pattern_results = sweep_pattern(detector, dataset, pattern_list, use_csi=args.use_csi)
        results_dict['pattern'] = pattern_results
        for r in pattern_results:
            all_results.append({**r, 'sweep_type': 'pattern'})
    
    if doppler_scale_list:
        doppler_results = sweep_doppler_scale(detector, dataset, doppler_scale_list, use_csi=args.use_csi)
        results_dict['doppler'] = doppler_results
        for r in doppler_results:
            all_results.append({**r, 'sweep_type': 'doppler'})
    
    # Save results to CSV
    scenario_name = 'scenario_a' if args.scenario == 'sat' else 'scenario_b'
    output_csv = args.output_csv or f"{RESULT_DIR}/robustness_{scenario_name}.csv"
    output_dir = args.output_dir or RESULT_DIR
    
    if all_results:
        # Flatten results for CSV
        csv_rows = []
        for r in all_results:
            row = {
                'sweep_type': r.get('sweep_type', ''),
                'auc': r.get('auc', np.nan) if r.get('auc') is not None else np.nan,
                'precision': r.get('precision', np.nan) if r.get('precision') is not None else np.nan,
                'recall': r.get('recall', np.nan) if r.get('recall') is not None else np.nan,
                'f1': r.get('f1', np.nan) if r.get('f1') is not None else np.nan,
                'threshold': r.get('threshold', np.nan) if r.get('threshold') is not None else np.nan,
                'n_samples': r.get('n_samples', 0)
            }
            
            # Add parameter value
            if 'snr_db' in r:
                row['snr_db'] = r['snr_db']
            if 'covert_amp' in r:
                row['covert_amp'] = r['covert_amp']
                # ✅ FILTER: Skip COVERT_AMP < 0.3 in main results
                if r['covert_amp'] < 0.3:
                    continue  # Skip this row
            if 'pattern' in r:
                row['pattern'] = r['pattern']
            if 'doppler_scale' in r:
                row['doppler_scale'] = r['doppler_scale']
            if 'threshold' in r:
                row['threshold'] = r['threshold']
            if 'class_counts' in r:
                row['class_counts'] = r['class_counts']

            csv_rows.append(row)
        
        df = pd.DataFrame(csv_rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
    
    # Generate plots
    if results_dict:
        plot_robustness_results(results_dict, output_dir, scenario_name)
    
    print("\n" + "="*70)
    print("✅ ROBUSTNESS ANALYSIS COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

