#!/usr/bin/env python3
"""
🔧 Phase 5: CSI Quality Analysis
=================================
Analyzes CSI quality and its impact on detection performance.

Generates:
- AUC vs NMSE(CSI) plot
- NMSE histogram
- Quality distribution analysis
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config.settings import GLOBAL_SEED, RESULT_DIR, DATASET_DIR
from utils.reproducibility import set_global_seeds, log_seed_info
from model.detector_cnn import CNNDetector

# 🔒 Phase 0: Set global seeds
log_seed_info("analyze_csi.py")
set_global_seeds(deterministic=True)


def analyze_csi_quality(dataset_path, scenario, bins=20):
    """
    Analyze CSI quality and its impact on detection.
    
    Args:
        dataset_path: Path to dataset file
        scenario: 'sat' or 'ground'
        bins: Number of bins for histogram
    """
    print("="*70)
    print("🔧 PHASE 5: CSI QUALITY ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Scenario: {scenario}")
    print("="*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    if 'csi_est' not in dataset or dataset['csi_est'] is None:
        print("❌ No CSI data found in dataset")
        return False
    
    X_csi = dataset['csi_est']
    
    print(f"✓ Dataset loaded: {len(Y)} samples")
    print(f"✓ CSI shape: {X_csi.shape}")
    
    # Extract CSI quality metrics from metadata
    csi_nmse_list = []
    csi_variance_list = []
    csi_quality_list = []
    
    if 'meta' in dataset and dataset['meta']:
        for meta in dataset['meta']:
            if isinstance(meta, dict):
                csi_nmse_list.append(meta.get('csi_nmse_db'))
                csi_variance_list.append(meta.get('csi_variance'))
                csi_quality_list.append(meta.get('csi_quality'))
    
    # If NMSE not in metadata, compute from CSI variance
    if all(nmse is None for nmse in csi_nmse_list):
        print("  ⚠️  NMSE not in metadata, computing from CSI variance...")
        for i in range(len(X_csi)):
            h_est = X_csi[i]
            h_mag = np.abs(h_est)
            variance = np.var(h_mag)
            csi_variance_list.append(variance)
            # Approximate NMSE from variance (higher variance = worse quality)
            # This is a proxy since we don't have true CSI
            csi_nmse_list.append(10.0 * np.log10(variance + 1e-12))
    
    csi_nmse_array = np.array([nmse if nmse is not None else np.nan for nmse in csi_nmse_list])
    csi_variance_array = np.array([var if var is not None else np.nan for var in csi_variance_list])
    
    # Remove NaN values
    valid_mask = ~np.isnan(csi_nmse_array)
    csi_nmse_valid = csi_nmse_array[valid_mask]
    
    print(f"\n📊 CSI Quality Statistics:")
    print(f"  NMSE (dB): mean={np.mean(csi_nmse_valid):.2f}, std={np.std(csi_nmse_valid):.2f}")
    print(f"  Variance: mean={np.mean(csi_variance_array[valid_mask]):.6e}, std={np.std(csi_variance_array[valid_mask]):.6e}")
    
    # Split data
    X_train, X_test, X_csi_train, X_csi_test, y_train, y_test = train_test_split(
        X_grids, X_csi, Y,
        test_size=0.3,
        random_state=GLOBAL_SEED,
        stratify=Y
    )
    
    X_tr, X_val, X_csi_tr, X_csi_val, y_tr, y_val = train_test_split(
        X_train, X_csi_train, y_train,
        test_size=0.2/0.7,
        random_state=GLOBAL_SEED,
        stratify=y_train
    )
    
    # Get test set NMSE
    # Create index mapping for test set
    all_indices = np.arange(len(Y))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.3,
        random_state=GLOBAL_SEED,
        stratify=Y
    )
    csi_nmse_test = csi_nmse_array[test_indices]
    
    # Train CNN+CSI model
    print(f"\n🤖 Training CNN+CSI model...")
    detector = CNNDetector(
        use_csi=True,
        learning_rate=0.001,
        dropout_rate=0.3,
        random_state=GLOBAL_SEED,
        use_focal_loss=True,
        focal_gamma=2.5,
        focal_alpha=0.5
    )
    
    detector.train(
        X_tr, y_tr,
        X_csi_train=X_csi_tr,
        X_val=X_val,
        y_val=y_val,
        X_csi_val=X_csi_val,
        epochs=50,
        batch_size=512,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    results = detector.evaluate(X_test, y_test, X_csi_test=X_csi_test)
    
    y_proba = detector.predict_proba(X_test, X_csi_test=X_csi_test)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"  ✓ Test AUC: {auc:.4f}")
    print(f"  ✓ Precision: {results['precision']:.4f}")
    print(f"  ✓ Recall: {results['recall']:.4f}")
    print(f"  ✓ F1: {results['f1']:.4f}")
    
    # Analyze AUC vs NMSE
    print(f"\n📈 Analyzing AUC vs CSI Quality...")
    
    # Bin NMSE values and compute AUC per bin
    if len(csi_nmse_test) > 0:
        nmse_min = np.min(csi_nmse_test)
        nmse_max = np.max(csi_nmse_test)
        nmse_bins = np.linspace(nmse_min, nmse_max, bins+1)
        
        bin_aucs = []
        bin_centers = []
        bin_counts = []
        
        for i in range(len(nmse_bins)-1):
            bin_mask = (csi_nmse_test >= nmse_bins[i]) & (csi_nmse_test < nmse_bins[i+1])
            if i == len(nmse_bins)-2:  # Include last bin
                bin_mask = (csi_nmse_test >= nmse_bins[i])
            
            if np.sum(bin_mask) > 10:  # At least 10 samples
                y_test_bin = y_test[bin_mask]
                y_proba_bin = y_proba[bin_mask]
                
                if len(np.unique(y_test_bin)) > 1:
                    bin_auc = roc_auc_score(y_test_bin, y_proba_bin)
                    bin_aucs.append(bin_auc)
                    bin_centers.append((nmse_bins[i] + nmse_bins[i+1]) / 2)
                    bin_counts.append(np.sum(bin_mask))
        
        # Plot AUC vs NMSE
        scenario_name = 'scenario_a' if scenario == 'sat' else 'scenario_b'
        output_dir = RESULT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, bin_aucs, 'o-', linewidth=2, markersize=8, label='AUC per NMSE bin')
        plt.axhline(y=auc, color='r', linestyle='--', label=f'Overall AUC={auc:.4f}')
        plt.xlabel('CSI NMSE (dB)', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.title(f'Detection Performance vs CSI Quality ({scenario_name})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/csi_auc_vs_nmse_{scenario_name}.png", dpi=150)
        plt.close()
        print(f"  ✓ Saved: csi_auc_vs_nmse_{scenario_name}.png")
        
        # Plot NMSE histogram
        plt.figure(figsize=(10, 6))
        plt.hist(csi_nmse_valid, bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel('CSI NMSE (dB)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'CSI Quality Distribution ({scenario_name})', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axvline(x=np.mean(csi_nmse_valid), color='r', linestyle='--', 
                   label=f'Mean={np.mean(csi_nmse_valid):.2f} dB')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/csi_nmse_histogram_{scenario_name}.png", dpi=150)
        plt.close()
        print(f"  ✓ Saved: csi_nmse_histogram_{scenario_name}.png")
        
        # Save analysis results
        analysis_results = {
            'overall_auc': float(auc),
            'overall_precision': float(results['precision']),
            'overall_recall': float(results['recall']),
            'overall_f1': float(results['f1']),
            'nmse_mean': float(np.mean(csi_nmse_valid)),
            'nmse_std': float(np.std(csi_nmse_valid)),
            'nmse_min': float(np.min(csi_nmse_valid)),
            'nmse_max': float(np.max(csi_nmse_valid)),
            'variance_mean': float(np.mean(csi_variance_array[valid_mask])),
            'variance_std': float(np.std(csi_variance_array[valid_mask]))
        }
        
        # Add bin-wise results
        if len(bin_aucs) > 0:
            df_bins = pd.DataFrame({
                'nmse_center': bin_centers,
                'auc': bin_aucs,
                'n_samples': bin_counts
            })
            bins_csv = f"{output_dir}/csi_auc_by_nmse_bin_{scenario_name}.csv"
            df_bins.to_csv(bins_csv, index=False)
            print(f"  ✓ Saved: csi_auc_by_nmse_bin_{scenario_name}.csv")
        
        summary_csv = f"{output_dir}/csi_analysis_summary_{scenario_name}.csv"
        df_summary = pd.DataFrame([analysis_results])
        df_summary.to_csv(summary_csv, index=False)
        print(f"  ✓ Saved: csi_analysis_summary_{scenario_name}.csv")
        
        # Conclusion
        print(f"\n{'='*70}")
        print("📊 ANALYSIS CONCLUSION")
        print(f"{'='*70}")
        
        if auc < 0.7:
            print("  ⚠️  CSI fusion degrades performance (AUC < 0.7)")
            print("  → Recommendation: Remove CSI fusion or improve CSI quality")
        elif auc < 0.85:
            print("  ⚠️  CSI fusion provides limited benefit")
            print("  → Recommendation: Improve CSI estimation or use quality gating")
        else:
            print("  ✓ CSI fusion works well (AUC ≥ 0.85)")
            print("  → Recommendation: Keep CSI fusion")
        
        if np.mean(csi_nmse_valid) > 0:
            print(f"  → Mean NMSE: {np.mean(csi_nmse_valid):.2f} dB (target: < -10 dB for good quality)")
    
    return True


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Phase 5: CSI quality analysis")
    parser.add_argument('--scenario', type=str, choices=['sat', 'ground'], required=True,
                       help="Scenario: 'sat' (scenario_a) or 'ground' (scenario_b)")
    parser.add_argument('--dataset', type=str, default=None,
                       help="Path to dataset file (auto-detect if not provided)")
    parser.add_argument('--bins', type=int, default=20,
                       help="Number of bins for histogram (default: 20)")
    
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
    
    # Run analysis
    success = analyze_csi_quality(args.dataset, args.scenario, bins=args.bins)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

