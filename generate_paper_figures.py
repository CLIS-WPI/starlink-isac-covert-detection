#!/usr/bin/env python3
"""
📊 Generate Paper Figures
==========================
Generates figures for the paper:
- Fig. 3: OFDM Resource Grid Heatmap (benign vs attack)
- Fig. 4: ROC Curves (Scenario A vs B)
- Fig. 5: Power Spectrum (benign vs attack)
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc

# Set matplotlib backend
matplotlib.use('Agg')  # Non-interactive backend

# Try to import tensorflow (optional for some figures)
try:
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. ROC curves will be skipped.")

from config.settings import DATASET_DIR, MODEL_DIR, RESULT_DIR
from model.detector_cnn import CNNDetector

# Set style
plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14


def find_latest_dataset(scenario='a'):
    """Find latest dataset for scenario."""
    pattern = f"{DATASET_DIR}/dataset_scenario_{scenario}*.pkl"
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def load_detector(scenario='a', use_csi=False):
    """Load trained CNN detector (with proper preprocessing)."""
    if not TF_AVAILABLE:
        return None
    
    scenario_folder = 'scenario_a' if scenario == 'a' else 'scenario_b'
    suffix = '_csi' if use_csi else ''
    model_path = f"{MODEL_DIR}/{scenario_folder}/cnn_detector{suffix}.keras"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return None
    
    try:
        # Load as CNNDetector to get proper preprocessing
        detector = CNNDetector(use_csi=use_csi)
        detector.model = tf.keras.models.load_model(model_path)
        detector.is_trained = True
        print(f"✅ Loaded detector: {model_path}")
        return detector
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return None


def generate_fig3_resource_grid(dataset_path, output_path='result/fig3_resource_grid.png'):
    """
    Fig. 3: OFDM Resource Grid Heatmap
    Shows benign vs attack samples with covert pattern visible.
    """
    print("\n" + "="*70)
    print("📊 Generating Fig. 3: OFDM Resource Grid Heatmap")
    print("="*70)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    rx_grids = dataset['rx_grids']
    labels = dataset['labels']
    
    # Find benign and attack samples
    benign_idx = np.where(labels == 0)[0]
    attack_idx = np.where(labels == 1)[0]
    
    if len(benign_idx) == 0 or len(attack_idx) == 0:
        print("❌ No benign or attack samples found")
        return False
    
    # Select samples (prefer samples with middle band injection)
    benign_sample = rx_grids[benign_idx[0]]
    attack_sample = None
    
    # Try to find attack sample with middle band injection (subcarriers 24-39)
    for idx in attack_idx[:100]:  # Check first 100 attack samples
        sample = rx_grids[idx]
        # Check if there's pattern in middle band
        mid_band_power = np.mean(np.abs(sample[:, 24:40]))
        if mid_band_power > np.mean(np.abs(sample)):
            attack_sample = sample
            break
    
    if attack_sample is None:
        attack_sample = rx_grids[attack_idx[0]]
    
    # Convert to magnitude
    benign_mag = np.abs(benign_sample)
    attack_mag = np.abs(attack_sample)
    
    # Normalize to [0, 1]
    benign_mag_norm = (benign_mag - benign_mag.min()) / (benign_mag.max() - benign_mag.min() + 1e-10)
    attack_mag_norm = (attack_mag - attack_mag.min()) / (attack_mag.max() - attack_mag.min() + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot benign
    im1 = axes[0].imshow(benign_mag_norm.T, aspect='auto', origin='lower', 
                        cmap='viridis', interpolation='nearest')
    axes[0].set_title('Benign Sample', fontweight='bold')
    axes[0].set_xlabel('OFDM Symbol Index')
    axes[0].set_ylabel('Subcarrier Index')
    axes[0].axhspan(24, 39, alpha=0.2, color='red', label='Covert Band (24-39)')
    plt.colorbar(im1, ax=axes[0], label='Normalized Magnitude')
    
    # Plot attack
    im2 = axes[1].imshow(attack_mag_norm.T, aspect='auto', origin='lower', 
                        cmap='viridis', interpolation='nearest')
    axes[1].set_title('Attack Sample (Covert Pattern)', fontweight='bold')
    axes[1].set_xlabel('OFDM Symbol Index')
    axes[1].set_ylabel('Subcarrier Index')
    axes[1].axhspan(24, 39, alpha=0.3, color='red', label='Covert Band (24-39)')
    plt.colorbar(im2, ax=axes[1], label='Normalized Magnitude')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return True


def generate_fig4_roc_curves(output_path='result/fig4_roc_curves.png'):
    """
    Fig. 4: ROC Curves for Scenario A and B
    """
    print("\n" + "="*70)
    print("📊 Generating Fig. 4: ROC Curves")
    print("="*70)
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available. Skipping ROC curves.")
        return False
    
    results = {}
    
    # Load results for both scenarios
    for scenario in ['a', 'b']:
        scenario_folder = 'scenario_a' if scenario == 'a' else 'scenario_b'
        result_path = f"{RESULT_DIR}/{scenario_folder}/detection_results_cnn.json"
        
        if not os.path.exists(result_path):
            print(f"⚠️  Results not found: {result_path}")
            continue
        
        # Load dataset and model to compute ROC
        dataset_path = find_latest_dataset(scenario)
        if not dataset_path:
            print(f"⚠️  Dataset not found for scenario {scenario}")
            continue
        
        detector = load_detector(scenario, use_csi=False)
        if detector is None:
            continue
        
        # Load dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        rx_grids = dataset['rx_grids']
        labels = dataset['labels']
        
        # Split data (same as training)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42, stratify=labels)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])
        
        # Get test data
        X_test = rx_grids[test_idx]
        y_test = labels[test_idx]
        
        # Use detector's predict_proba (handles preprocessing correctly)
        y_score = detector.predict_proba(X_test)
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        results[scenario] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        
        print(f"✅ Scenario {scenario.upper()}: AUC = {roc_auc:.4f}")
    
    if len(results) == 0:
        print("❌ No results found")
        return False
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'a': '#2E86AB', 'b': '#A23B72'}
    labels_map = {'a': 'Scenario A (Direct Link)', 'b': 'Scenario B (Dual-Hop + MMSE)'}
    
    for scenario in ['a', 'b']:
        if scenario in results:
            r = results[scenario]
            ax.plot(r['fpr'], r['tpr'], 
                   color=colors[scenario],
                   lw=2.5,
                   label=f"{labels_map[scenario]} (AUC = {r['auc']:.4f})")
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves for Covert Leakage Detection', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return True


def generate_fig5_power_spectrum(dataset_path, output_path='result/fig5_power_spectrum.png'):
    """
    Fig. 5: Power Spectrum Comparison (Benign vs Attack)
    Shows average power spectrum highlighting covert subcarriers.
    """
    print("\n" + "="*70)
    print("📊 Generating Fig. 5: Power Spectrum")
    print("="*70)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    rx_grids = dataset['rx_grids']
    labels = dataset['labels']
    
    # Separate benign and attack
    benign_samples = rx_grids[labels == 0]
    attack_samples = rx_grids[labels == 1]
    
    if len(benign_samples) == 0 or len(attack_samples) == 0:
        print("❌ No benign or attack samples found")
        return False
    
    # Compute average power spectrum (across symbols)
    benign_power = np.mean(np.abs(benign_samples), axis=(0, 1))  # Average over samples and symbols
    attack_power = np.mean(np.abs(attack_samples), axis=(0, 1))
    
    # Subcarrier indices
    num_subcarriers = len(benign_power)
    subcarrier_idx = np.arange(num_subcarriers)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot power spectrum
    ax.plot(subcarrier_idx, benign_power, 'b-', lw=2, label='Benign (Average)', alpha=0.7)
    ax.plot(subcarrier_idx, attack_power, 'r-', lw=2, label='Attack (Average)', alpha=0.7)
    
    # Highlight covert band (24-39)
    ax.axvspan(24, 39, alpha=0.2, color='orange', label='Covert Band (24-39)')
    
    # Mark injection region
    for sc in [24, 39]:
        ax.axvline(sc, color='orange', linestyle='--', alpha=0.5, lw=1)
    
    ax.set_xlabel('Subcarrier Index', fontweight='bold')
    ax.set_ylabel('Average Power Magnitude', fontweight='bold')
    ax.set_title('Power Spectrum: Benign vs Attack Samples', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, num_subcarriers-1])
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return True


def main():
    """Generate all figures."""
    print("="*70)
    print("📊 PAPER FIGURES GENERATOR")
    print("="*70)
    
    # Find latest datasets
    dataset_a = find_latest_dataset('a')
    dataset_b = find_latest_dataset('b')
    
    if dataset_a:
        print(f"✅ Found Scenario A dataset: {os.path.basename(dataset_a)}")
    else:
        print("⚠️  Scenario A dataset not found")
    
    if dataset_b:
        print(f"✅ Found Scenario B dataset: {os.path.basename(dataset_b)}")
    else:
        print("⚠️  Scenario B dataset not found")
    
    # Generate figures
    success_count = 0
    
    # Fig. 3: Resource Grid (use Scenario A for clarity)
    if dataset_a:
        if generate_fig3_resource_grid(dataset_a, 'result/fig3_resource_grid.png'):
            success_count += 1
    else:
        print("⚠️  Skipping Fig. 3 (no dataset)")
    
    # Fig. 4: ROC Curves
    if generate_fig4_roc_curves('result/fig4_roc_curves.png'):
        success_count += 1
    
    # Fig. 5: Power Spectrum (use Scenario A)
    if dataset_a:
        if generate_fig5_power_spectrum(dataset_a, 'result/fig5_power_spectrum.png'):
            success_count += 1
    else:
        print("⚠️  Skipping Fig. 5 (no dataset)")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    print(f"✅ Generated {success_count}/3 figures")
    print(f"📁 Output directory: result/")
    print("\nGenerated figures:")
    print("  - fig3_resource_grid.png: OFDM Resource Grid Heatmap")
    print("  - fig4_roc_curves.png: ROC Curves (Scenario A vs B)")
    print("  - fig5_power_spectrum.png: Power Spectrum Comparison")
    print("="*70)
    
    return success_count == 3


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

