#!/usr/bin/env python3
"""
🔧 Phase 4: Cross-Validation with Multiple Seeds
================================================
Performs K-fold cross-validation with N different seeds to assess model stability.

Usage:
    python3 run_cross_validation.py --scenario sat --kfold 5 --seeds 3
    python3 run_cross_validation.py --scenario ground --kfold 5 --seeds 3
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config.settings import (
    DATASET_DIR, MODEL_DIR, RESULT_DIR, GLOBAL_SEED, N_FOLDS, N_SEEDS
)
from utils.reproducibility import set_global_seeds, log_seed_info
from model.detector_cnn import CNNDetector

# 🔒 Phase 0: Set global seeds
log_seed_info("run_cross_validation.py")
set_global_seeds(deterministic=True)


def run_cross_validation(scenario, dataset_path, k_folds=5, n_seeds=3, 
                        use_csi=False, epochs=30, batch_size=512):
    """
    Run K-fold cross-validation with N seeds.
    
    Args:
        scenario: 'sat' (scenario_a) or 'ground' (scenario_b)
        dataset_path: Path to dataset file
        k_folds: Number of folds (default: 5)
        n_seeds: Number of random seeds (default: 3)
        use_csi: Whether to use CNN+CSI (default: False)
        epochs: Number of training epochs (default: 30 for CV)
        batch_size: Batch size (default: 512)
    
    Returns:
        dict: Summary statistics (mean±std) for all metrics
    """
    print("="*70)
    print(f"🔧 PHASE 4: CROSS-VALIDATION ({k_folds}-fold, {n_seeds} seeds)")
    print("="*70)
    print(f"Scenario: {scenario}")
    print(f"Model: {'CNN+CSI' if use_csi else 'CNN-only'}")
    print(f"K-folds: {k_folds}")
    print(f"Seeds: {n_seeds}")
    print(f"Epochs per fold: {epochs}")
    print("="*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    if use_csi and 'csi_est' in dataset and dataset['csi_est'] is not None:
        X_csi = dataset['csi_est']
    else:
        X_csi = None
        use_csi = False  # Override if CSI not available
    
    print(f"✓ Dataset loaded: {len(Y)} samples")
    print(f"✓ Benign: {np.sum(Y==0)}, Attack: {np.sum(Y==1)}")
    
    # Collect all results
    all_results = []
    
    # Loop over seeds
    base_seed = GLOBAL_SEED
    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx * 1000
        print(f"\n{'='*70}")
        print(f"🌱 SEED {seed_idx+1}/{n_seeds} (seed={seed})")
        print(f"{'='*70}")
        
        # Set seed for this run
        np.random.seed(seed)
        import random
        random.seed(seed)
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # K-fold cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_grids, Y)):
            print(f"\n  📊 Fold {fold_idx+1}/{k_folds}")
            print(f"     Train: {len(train_idx)}, Val: {len(val_idx)}")
            
            # Split data
            X_train = X_grids[train_idx]
            X_val = X_grids[val_idx]
            y_train = Y[train_idx]
            y_val = Y[val_idx]
            
            X_csi_train = X_csi[train_idx] if X_csi is not None else None
            X_csi_val = X_csi[val_idx] if X_csi is not None else None
            
            # Create and train detector
            detector = CNNDetector(
                use_csi=use_csi,
                learning_rate=0.001,
                dropout_rate=0.3,
                random_state=seed,
                use_focal_loss=True,
                focal_gamma=2.5,
                focal_alpha=0.5
            )
            
            # Train with early stopping (reduced epochs for CV)
            print(f"     Training...")
            detector.train(
                X_train, y_train,
                X_csi_train=X_csi_train,
                X_val=X_val,
                y_val=y_val,
                X_csi_val=X_csi_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0  # Reduce output
            )
            
            # Evaluate on validation set
            results = detector.evaluate(X_val, y_val, X_csi_test=X_csi_val)
            
            fold_result = {
                'seed': seed,
                'seed_idx': seed_idx,
                'fold': fold_idx,
                'auc': results.get('auc', 0.0),
                'precision': results.get('precision', 0.0),
                'recall': results.get('recall', 0.0),
                'f1': results.get('f1', 0.0),
                'threshold': results.get('threshold', 0.5)
            }
            fold_results.append(fold_result)
            all_results.append(fold_result)
            
            print(f"     ✓ Fold {fold_idx+1} Results:")
            print(f"        AUC: {fold_result['auc']:.4f}, F1: {fold_result['f1']:.4f}")
        
        # Summary for this seed
        seed_aucs = [r['auc'] for r in fold_results]
        seed_f1s = [r['f1'] for r in fold_results]
        print(f"\n  📊 Seed {seed_idx+1} Summary:")
        print(f"     AUC: {np.mean(seed_aucs):.4f} ± {np.std(seed_aucs):.4f}")
        print(f"     F1:  {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
    
    # Compute overall statistics
    print(f"\n{'='*70}")
    print("📊 OVERALL CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    metrics = ['auc', 'precision', 'recall', 'f1', 'threshold']
    summary = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        print(f"{metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f} "
              f"(range: [{np.min(values):.4f}, {np.max(values):.4f}])")
    
    # Save detailed results
    df_detailed = pd.DataFrame(all_results)
    scenario_name = 'scenario_a' if scenario == 'sat' else 'scenario_b'
    model_suffix = '_csi' if use_csi else ''
    detailed_csv = f"{RESULT_DIR}/cv_detailed_{scenario_name}{model_suffix}.csv"
    os.makedirs(os.path.dirname(detailed_csv), exist_ok=True)
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\n✓ Detailed results saved to: {detailed_csv}")
    
    # Save summary
    summary_rows = []
    for metric in metrics:
        summary_rows.append({
            'metric': metric,
            'mean': summary[metric]['mean'],
            'std': summary[metric]['std'],
            'min': summary[metric]['min'],
            'max': summary[metric]['max']
        })
    
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = f"{RESULT_DIR}/cv_summary_{scenario_name}{model_suffix}.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Summary saved to: {summary_csv}")
    
    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Phase 4: Cross-validation with multiple seeds")
    parser.add_argument('--scenario', type=str, choices=['sat', 'ground'], required=True,
                       help="Scenario: 'sat' (scenario_a) or 'ground' (scenario_b)")
    parser.add_argument('--dataset', type=str, default=None,
                       help="Path to dataset file (auto-detect if not provided)")
    parser.add_argument('--kfold', type=int, default=5,
                       help="Number of folds (default: 5)")
    parser.add_argument('--seeds', type=int, default=3,
                       help="Number of random seeds (default: 3)")
    parser.add_argument('--use-csi', action='store_true',
                       help="Use CNN+CSI model (default: CNN-only)")
    parser.add_argument('--epochs', type=int, default=30,
                       help="Number of epochs per fold (default: 30 for CV)")
    parser.add_argument('--batch-size', type=int, default=512,
                       help="Batch size (default: 512)")
    
    args = parser.parse_args()
    
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
    
    # Run cross-validation
    summary = run_cross_validation(
        args.scenario,
        args.dataset,
        k_folds=args.kfold,
        n_seeds=args.seeds,
        use_csi=args.use_csi,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*70}")
    print("✅ CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

