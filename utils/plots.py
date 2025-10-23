# ======================================
# 📄 utils/plots.py
# Purpose: Centralized plotting utilities for all paper figures
# ======================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

sns.set(style='whitegrid')


def plot_roc_curve(y_te, y_prob, save_dir='result'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', lw=1, color='gray', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Covert Detection', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/1_roc_curve.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/1_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_dir}/1_roc_curve.pdf")


def plot_precision_recall_curve(y_te, y_prob, save_dir='result'):
    """Plot Precision-Recall curve."""
    p, r, _ = precision_recall_curve(y_te, y_prob)
    ap = average_precision_score(y_te, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(r, p, lw=2, label=f'AP={ap:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/2_precision_recall_curve.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/2_precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved to {save_dir}/2_precision_recall_curve.pdf")


def plot_confusion_matrix(y_te, y_hat, save_dir='result'):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_te, y_hat)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3_confusion_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/3_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion Matrix saved to {save_dir}/3_confusion_matrix.pdf")


def plot_training_history(hist, save_dir='result'):
    """Plot training history (accuracy and loss)."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(hist.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Model Accuracy', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('Model Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/5_training_history.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/5_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training History saved to {save_dir}/5_training_history.pdf")


def plot_power_spectrum(dataset, labels, isac_system, save_dir='result'):
    """Plot power spectrum comparison (benign vs attack)."""
    benign_samples = dataset['iq_samples'][labels == 0][:10]
    attack_samples = dataset['iq_samples'][labels == 1][:10]
    
    benign_fft = np.mean([np.abs(np.fft.fft(s))**2 for s in benign_samples], axis=0)
    attack_fft = np.mean([np.abs(np.fft.fft(s))**2 for s in attack_samples], axis=0)
    
    freq = np.fft.fftfreq(len(benign_fft), d=1/isac_system.SAMPLING_RATE) / 1e6  # MHz
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:len(freq)//2], 10*np.log10(benign_fft[:len(freq)//2] + 1e-12),
             label='Benign', linewidth=2, alpha=0.8)
    plt.plot(freq[:len(freq)//2], 10*np.log10(attack_fft[:len(freq)//2] + 1e-12),
             label='Attack (Covert)', linewidth=2, alpha=0.8)
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Power Spectrum (dB)', fontsize=12)
    plt.title('Power Spectrum: Benign vs Attack', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/6_power_spectrum.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/6_power_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Power Spectrum saved to {save_dir}/6_power_spectrum.pdf")


def plot_f1_vs_threshold(f1_scores, t, best_thr, best_idx, save_dir='result'):
    """Plot F1 score vs threshold."""
    plt.figure(figsize=(8, 6))
    plt.plot(t, f1_scores[:-1], linewidth=2, label='F1 Score')
    plt.axvline(best_thr, color='r', linestyle='--', linewidth=2,
                label=f'Best Threshold={best_thr:.3f}')
    plt.axhline(f1_scores[best_idx], color='g', linestyle=':', linewidth=1, alpha=0.5)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/7_f1_vs_threshold.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/7_f1_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ F1 vs Threshold saved to {save_dir}/7_f1_vs_threshold.pdf")


def plot_localization_cdf(loc_errors, save_dir='result'):
    """Plot localization error CDF with safe markevery."""
    if not loc_errors:
        return
    
    med = float(np.median(loc_errors))
    p90 = float(np.percentile(loc_errors, 90))
    
    xs = np.sort(loc_errors)
    cdf = np.arange(1, len(xs)+1)/len(xs)
    
    # ✅ FIX: Safe markevery computation
    mark_step = max(1, len(xs)//20)  # Ensure step >= 1
    
    plt.figure(figsize=(8, 6))
    plt.plot(xs, cdf, linewidth=2, marker='o', markersize=3, markevery=mark_step)
    plt.axhline(0.9, color='r', ls='--', linewidth=2, 
                label=f'90th percentile = {p90:.1f} m')
    plt.axhline(0.5, color='orange', ls=':', linewidth=1, alpha=0.5, 
                label=f'Median = {med:.1f} m')
    plt.xlabel('Localization Error (m)', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Localization Error CDF', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/4_localization_cdf.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/4_localization_cdf.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Localization CDF saved to {save_dir}/4_localization_cdf.pdf")


def plot_localization_histogram(loc_errors, save_dir='result'):
    """Plot localization error histogram."""
    if not loc_errors:
        return
    
    med = float(np.median(loc_errors))
    mean = float(np.mean(loc_errors))
    
    plt.figure(figsize=(8, 6))
    plt.hist(loc_errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(med, color='r', linestyle='--', linewidth=2, label=f'Median={med:.1f}m')
    plt.axvline(mean, color='g', linestyle=':', linewidth=2, label=f'Mean={mean:.1f}m')
    plt.xlabel('Localization Error (m)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Localization Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/8_localization_histogram.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/8_localization_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Localization Histogram saved to {save_dir}/8_localization_histogram.pdf")


def plot_spectrogram_examples(Xs_te, y_te, y_hat, save_dir='result'):
    """Plot 2x2 grid of spectrogram examples."""
    benign_correct = np.where((y_te == 0) & (y_hat == 0))[0]
    benign_wrong = np.where((y_te == 0) & (y_hat == 1))[0]
    attack_correct = np.where((y_te == 1) & (y_hat == 1))[0]
    attack_wrong = np.where((y_te == 1) & (y_hat == 0))[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if len(benign_correct) > 0:
        idx = benign_correct[0]
        spec = Xs_te[idx].squeeze()
        axes[0, 0].imshow(spec, aspect='auto', cmap='viridis', origin='lower')
        axes[0, 0].set_title('Benign - Correctly Detected', fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
    
    if len(benign_wrong) > 0:
        idx = benign_wrong[0]
        spec = Xs_te[idx].squeeze()
        axes[0, 1].imshow(spec, aspect='auto', cmap='viridis', origin='lower')
        axes[0, 1].set_title('Benign - Misclassified as Attack', 
                            fontweight='bold', color='red')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
    
    if len(attack_correct) > 0:
        idx = attack_correct[0]
        spec = Xs_te[idx].squeeze()
        axes[1, 0].imshow(spec, aspect='auto', cmap='plasma', origin='lower')
        axes[1, 0].set_title('Attack - Correctly Detected', fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
    
    if len(attack_wrong) > 0:
        idx = attack_wrong[0]
        spec = Xs_te[idx].squeeze()
        axes[1, 1].imshow(spec, aspect='auto', cmap='plasma', origin='lower')
        axes[1, 1].set_title('Attack - Missed Detection', 
                            fontweight='bold', color='red')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/9_spectrogram_examples.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/9_spectrogram_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Spectrogram Examples saved to {save_dir}/9_spectrogram_examples.pdf")


def plot_satellite_geometry(dataset, save_dir='result'):
    """Plot 3D + 2D satellite constellation geometry."""
    if 'satellite_receptions' not in dataset or len(dataset['satellite_receptions']) == 0:
        return
    
    first_sample_sats = dataset['satellite_receptions'][0]
    sat_positions = np.array([s['position'] for s in first_sample_sats])
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(sat_positions[:, 0]/1e3, sat_positions[:, 1]/1e3,
               sat_positions[:, 2]/1e3, s=200, c='red', marker='^',
               edgecolors='black', linewidths=2, label='Satellites')
    ax1.scatter([0], [0], [0], s=300, c='blue', marker='o',
               edgecolors='black', linewidths=2, label='Ground (0,0,0)')
    
    for i, pos in enumerate(sat_positions):
        ax1.text(pos[0]/1e3, pos[1]/1e3, pos[2]/1e3, f'  S{i+1}', fontsize=9)
    
    ax1.set_xlabel('X (km)', fontsize=11)
    ax1.set_ylabel('Y (km)', fontsize=11)
    ax1.set_zlabel('Altitude (km)', fontsize=11)
    ax1.set_title('3D Satellite Constellation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2D top view
    ax2 = fig.add_subplot(122)
    sc = ax2.scatter(sat_positions[:, 0]/1e3, sat_positions[:, 1]/1e3,
                    s=200, c=sat_positions[:, 2]/1e3, cmap='viridis',
                    marker='^', edgecolors='black', linewidths=2)
    cbar = plt.colorbar(sc, ax=ax2, label='Altitude (km)')
    
    for i, pos in enumerate(sat_positions):
        ax2.text(pos[0]/1e3 + 5, pos[1]/1e3 + 5, 
                f'S{i+1}\n{pos[2]/1e3:.1f}km',
                fontsize=9, ha='left')
    
    ax2.scatter([0], [0], s=300, c='red', marker='x', linewidths=3, label='Ground (0,0)')
    ax2.set_xlabel('X (km)', fontsize=11)
    ax2.set_ylabel('Y (km)', fontsize=11)
    ax2.set_title('Top View (with Altitude Diversity)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/10_satellite_geometry.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/10_satellite_geometry.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Satellite Geometry saved to {save_dir}/10_satellite_geometry.pdf")


def plot_calibration_curve(y_te, y_prob, save_dir='result'):
    """Plot calibration curve with ECE and Brier score."""
    try:
        ece_bins = 10
        prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=ece_bins)
        brier = brier_score_loss(y_te, y_prob)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 'o-', label=f'Model (ECE={ece:.3f})')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/11_calibration_curve.pdf", dpi=300)
        plt.close()
        
        print(f"\n=== CALIBRATION METRICS ===")
        print(f"Brier Score: {brier:.4f}")
        print(f"ECE (Expected Calibration Error): {ece:.4f}")
        print(f"✓ Calibration curve saved")
    except Exception as e:
        print(f"⚠️ Calibration metrics skipped: {e}")


def generate_all_plots(y_te, y_hat, y_prob, Xs_te, dataset, hist,
                      best_thr, best_idx, f1_scores, t,
                      loc_errors, isac_system, save_dir='result'):
    """Generate all paper plots in one function."""
    print("\n=== GENERATING PAPER PLOTS ===")
    
    # Classification metrics
    plot_roc_curve(y_te, y_prob, save_dir)
    plot_precision_recall_curve(y_te, y_prob, save_dir)
    plot_confusion_matrix(y_te, y_hat, save_dir)
    plot_training_history(hist, save_dir)
    plot_calibration_curve(y_te, y_prob, save_dir)
    
    # Signal analysis
    plot_power_spectrum(dataset, dataset['labels'], isac_system, save_dir)
    plot_f1_vs_threshold(f1_scores, t, best_thr, best_idx, save_dir)
    plot_spectrogram_examples(Xs_te, y_te, y_hat, save_dir)
    
    # Localization
    if loc_errors:
        plot_localization_cdf(loc_errors, save_dir)
        plot_localization_histogram(loc_errors, save_dir)
    
    # Geometry
    plot_satellite_geometry(dataset, save_dir)
    
    print("✓ All plots generated successfully")