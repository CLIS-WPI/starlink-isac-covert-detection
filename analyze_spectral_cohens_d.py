#!/usr/bin/env python3
"""
Calculate Cohen's d in spectral domain to verify covert detection feasibility
"""

import pickle
import numpy as np
from scipy import signal

def compute_spectrogram(iq_signal, fs=30.72e6, nperseg=256, noverlap=128):
    """Compute spectrogram using STFT"""
    f, t, Sxx = signal.spectrogram(iq_signal, fs=fs, nperseg=nperseg, 
                                     noverlap=noverlap, return_onesided=False)
    return np.abs(Sxx)

def analyze_spectral_separability(dataset_path):
    """Analyze separability in spectral domain"""
    print("=" * 60)
    print("SPECTRAL DOMAIN SEPARABILITY ANALYSIS")
    print("=" * 60)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    iq_samples = dataset['iq_samples']
    labels = dataset['labels']
    sampling_rate = dataset.get('sampling_rate', 30.72e6)
    
    # Separate benign and attack
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    attack_indices = [i for i, label in enumerate(labels) if label == 1]
    
    print(f"\n📊 Analyzing {min(100, len(benign_indices))} samples per class...")
    print("   (Processing first 100 samples for speed)")
    
    # Compute spectrograms
    benign_spectral_features = []
    attack_spectral_features = []
    
    print("\n🔄 Computing spectrograms...")
    
    # Use first 100 samples for speed
    for i in benign_indices[:100]:
        spec = compute_spectrogram(iq_samples[i], fs=sampling_rate)
        # Extract features: mean, std, max across time
        features = [
            np.mean(spec),
            np.std(spec),
            np.max(spec),
            np.median(spec),
            np.percentile(spec, 90),
        ]
        benign_spectral_features.append(features)
    
    for i in attack_indices[:100]:
        spec = compute_spectrogram(iq_samples[i], fs=sampling_rate)
        features = [
            np.mean(spec),
            np.std(spec),
            np.max(spec),
            np.median(spec),
            np.percentile(spec, 90),
        ]
        attack_spectral_features.append(features)
    
    benign_spectral_features = np.array(benign_spectral_features)
    attack_spectral_features = np.array(attack_spectral_features)
    
    print("✓ Spectrograms computed")
    
    # Calculate Cohen's d for each feature
    feature_names = ['Mean', 'Std', 'Max', 'Median', 'P90']
    
    print(f"\n📈 SPECTRAL COHEN'S D:")
    print("-" * 60)
    print(f"{'Feature':<15} {'Benign Mean':<15} {'Attack Mean':<15} {'Cohens_d':<10}")
    print("-" * 60)
    
    cohens_d_values = []
    for i, name in enumerate(feature_names):
        benign_vals = benign_spectral_features[:, i]
        attack_vals = attack_spectral_features[:, i]
        
        benign_mean = np.mean(benign_vals)
        attack_mean = np.mean(attack_vals)
        benign_std = np.std(benign_vals)
        attack_std = np.std(attack_vals)
        
        pooled_std = np.sqrt((benign_std**2 + attack_std**2) / 2)
        cohens_d = (attack_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
        cohens_d_values.append(abs(cohens_d))
        
        print(f"{name:<15} {benign_mean:<15.6e} {attack_mean:<15.6e} {cohens_d:<10.4f}")
    
    print("-" * 60)
    
    avg_cohens_d = np.mean(cohens_d_values)
    max_cohens_d = np.max(cohens_d_values)
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Average Cohen's d (spectral): {avg_cohens_d:.4f}")
    print(f"  Maximum Cohen's d (spectral): {max_cohens_d:.4f}")
    
    # Verdict
    print(f"\n{'=' * 60}")
    print("VERDICT:")
    if max_cohens_d >= 0.5:
        print("✅ EXCELLENT: Strong spectral separability (Cohen's d ≥ 0.5)")
        print("   → Detection with CNN/spectrogram should work well")
        print("   → Expected AUC: 0.80-0.90+")
    elif max_cohens_d >= 0.3:
        print("✅ GOOD: Moderate spectral separability (Cohen's d ≥ 0.3)")
        print("   → Detection possible with advanced features")
        print("   → Expected AUC: 0.70-0.80")
    elif max_cohens_d >= 0.2:
        print("⚠️  FAIR: Weak spectral separability (Cohen's d ≥ 0.2)")
        print("   → Detection challenging, may need more data")
        print("   → Expected AUC: 0.60-0.70")
    else:
        print("❌ POOR: Very weak spectral separability (Cohen's d < 0.2)")
        print("   → Detection very difficult, may need different approach")
        print("   → Expected AUC: 0.50-0.60")
    
    print(f"{'=' * 60}\n")
    
    # Recommendation
    print("💡 RECOMMENDATIONS:")
    if max_cohens_d < 0.3:
        print("  1. Consider increasing covert signal strength (higher ESNO)")
        print("  2. Try different injection strategy (phase/timing modulation)")
        print("  3. Use more sophisticated features (wavelet, cepstrum)")
        print("  4. Increase dataset size (3000+ samples per class)")
    else:
        print("  ✓ Current configuration should work well!")
        print("  ✓ Proceed with training on full dataset")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "dataset/dataset_samples1500_sats12.pkl"
    
    try:
        analyze_spectral_separability(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Usage: python3 analyze_spectral_cohens_d.py [dataset_path]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
