#!/usr/bin/env python3
"""
Comprehensive power and spectral analysis of dataset
"""

import pickle
import numpy as np
from scipy import signal
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA
import sys

def compute_spectrogram(iq_signal, fs=30.72e6, nperseg=256, noverlap=128):
    """Compute spectrogram using STFT"""
    f, t, Sxx = signal.spectrogram(iq_signal, fs=fs, nperseg=nperseg, 
                                     noverlap=noverlap, return_onesided=False)
    return np.abs(Sxx)

def analyze_power_and_spectral(dataset_path, check_spectral=False):
    """Analyze both power domain and spectral domain"""
    
    print("="*60)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("="*60)

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    labels = dataset['labels']
    iq_samples = dataset['iq_samples']
    sampling_rate = dataset.get('sampling_rate', 30.72e6)

    # ========================================
    # PART 1: POWER DOMAIN ANALYSIS
    # ========================================
    
    print("\n" + "="*60)
    print("PART 1: POWER DOMAIN ANALYSIS")
    print("="*60)
    
    # Analyze first 10 samples of each class
    print("\n📊 First 10 samples of each class:")
    print("-"*60)
    print(f"{'Index':<8} {'Label':<8} {'Power':<15} {'Mean Mag':<15}")
    print("-"*60)

    benign_powers = []
    attack_powers = []

    for i in range(min(20, len(labels))):
        power = np.mean(np.abs(iq_samples[i])**2)
        mean_mag = np.mean(np.abs(iq_samples[i]))
        label = labels[i]
        
        if label == 0:
            benign_powers.append(power)
            class_name = "BENIGN"
        else:
            attack_powers.append(power)
            class_name = "ATTACK"
        
        print(f"{i:<8} {class_name:<8} {power:.6e}  {mean_mag:.6e}")

    # Overall statistics
    print("\n📈 OVERALL POWER STATISTICS:")
    print("-"*60)

    all_benign = [np.mean(np.abs(iq_samples[i])**2) for i in range(len(labels)) if labels[i] == 0]
    all_attack = [np.mean(np.abs(iq_samples[i])**2) for i in range(len(labels)) if labels[i] == 1]

    benign_mean = np.mean(all_benign)
    benign_std = np.std(all_benign)
    attack_mean = np.mean(all_attack)
    attack_std = np.std(attack_powers)

    print(f"Benign samples ({len(all_benign)}):")
    print(f"  Mean power: {benign_mean:.6e}")
    print(f"  Std power:  {benign_std:.6e}")
    print(f"  CV (std/mean): {benign_std/benign_mean:.4f}")

    print(f"\nAttack samples ({len(all_attack)}):")
    print(f"  Mean power: {attack_mean:.6e}")
    print(f"  Std power:  {attack_std:.6e}")
    print(f"  CV (std/mean): {attack_std/attack_mean:.4f}")

    power_ratio = attack_mean/benign_mean
    print(f"\n🎯 Power ratio (attack/benign): {power_ratio:.4f}")
    
    # Cohen's d in power domain
    pooled_std = np.sqrt((benign_std**2 + attack_std**2) / 2)
    cohens_d_power = abs(attack_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
    print(f"🎯 Cohen's d (power domain): {cohens_d_power:.4f}")

    # Verdict
    print("\n" + "="*60)
    print("POWER DOMAIN VERDICT:")
    print("="*60)

    if 0.95 <= power_ratio <= 1.05:
        print("✅ EXCELLENT: Truly covert (power ratio ≈ 1.0)")
        print("   → Simple power detector cannot detect")
    elif 0.90 <= power_ratio <= 1.10:
        print("✅ GOOD: Mostly covert (power ratio close to 1.0)")
        print("   → Requires careful power analysis to detect")
    elif 0.80 <= power_ratio <= 1.20:
        print("⚠️  WARNING: Somewhat detectable by power")
        print("   → Experienced analyst may notice")
    else:
        print("❌ POOR: Easily detectable by power!")
        print("   → NOT truly covert")
        
    if cohens_d_power < 0.1:
        print("✅ Cohen's d < 0.1 (excellent stealth)")
    elif cohens_d_power < 0.2:
        print("✅ Cohen's d < 0.2 (good stealth)")
    else:
        print("⚠️  Cohen's d ≥ 0.2 (detectable)")
    
    # ========================================
    # PART 2: SPECTRAL DOMAIN ANALYSIS
    # ========================================
    
    if check_spectral:
        print("\n" + "="*60)
        print("PART 2: SPECTRAL DOMAIN ANALYSIS")
        print("="*60)
        print("Analyzing 100 samples per class (for speed)...\n")
        
        benign_indices = [i for i, label in enumerate(labels) if label == 0]
        attack_indices = [i for i, label in enumerate(labels) if label == 1]
        
        # Compute spectral features
        benign_spectral_features = []
        attack_spectral_features = []
        
        print("🔄 Computing spectrograms...")
        
        for i in benign_indices[:100]:
            spec = compute_spectrogram(iq_samples[i], fs=sampling_rate)
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
        
        print("✓ Spectrograms computed\n")
        
        # Calculate Cohen's d for each feature
        feature_names = ['Mean', 'Std', 'Max', 'Median', 'P90']
        
        print("📊 SPECTRAL COHEN'S D:")
        print("-" * 60)
        print(f"{'Feature':<15} {'Benign':<15} {'Attack':<15} {'Cohens_d':<10}")
        print("-" * 60)
        
        cohens_d_values = []
        for i, name in enumerate(feature_names):
            benign_vals = benign_spectral_features[:, i]
            attack_vals = attack_spectral_features[:, i]
            
            benign_mean_feat = np.mean(benign_vals)
            attack_mean_feat = np.mean(attack_vals)
            benign_std_feat = np.std(benign_vals)
            attack_std_feat = np.std(attack_vals)
            
            pooled_std_feat = np.sqrt((benign_std_feat**2 + attack_std_feat**2) / 2)
            cohens_d_feat = (attack_mean_feat - benign_mean_feat) / pooled_std_feat if pooled_std_feat > 0 else 0
            cohens_d_values.append(abs(cohens_d_feat))
            
            print(f"{name:<15} {benign_mean_feat:<15.6e} {attack_mean_feat:<15.6e} {cohens_d_feat:<10.4f}")
        
        print("-" * 60)
        
        avg_cohens_d = np.mean(cohens_d_values)
        max_cohens_d = np.max(cohens_d_values)
        
        print(f"\n🎯 Average Cohen's d (spectral): {avg_cohens_d:.4f}")
        print(f"🎯 Maximum Cohen's d (spectral): {max_cohens_d:.4f}")
        
        # Spectral verdict
        print("\n" + "="*60)
        print("SPECTRAL DOMAIN VERDICT:")
        print("="*60)
        
        if max_cohens_d >= 0.5:
            print("✅ EXCELLENT: Strong spectral separability")
            print("   → Detection with CNN should work well")
            print("   → Expected AUC: 0.80-0.90+")
        elif max_cohens_d >= 0.3:
            print("✅ GOOD: Moderate spectral separability")
            print("   → Detection possible with advanced features")
            print("   → Expected AUC: 0.70-0.80")
        elif max_cohens_d >= 0.2:
            print("⚠️  FAIR: Weak spectral separability")
            print("   → Detection challenging")
            print("   → Expected AUC: 0.60-0.70")
        else:
            print("❌ POOR: Very weak spectral separability")
            print("   → Detection very difficult")
            print("   → Expected AUC: 0.50-0.60")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if check_spectral:
        print(f"Power domain:   Ratio={power_ratio:.4f}, Cohen's d={cohens_d_power:.4f}")
        print(f"Spectral domain: Max Cohen's d={max_cohens_d:.4f}")
        print()
        
        if 0.95 <= power_ratio <= 1.05 and max_cohens_d >= 0.3:
            print("✅ IDEAL: Covert in power, detectable in spectral")
            print("   → Proceed with training!")
        elif power_ratio > 1.2:
            print("❌ NOT COVERT: Power too high")
            print("   → Apply power normalization")
        elif max_cohens_d < 0.2:
            print("⚠️  WEAK SIGNAL: Low spectral separability")
            print("   → Consider increasing ESNO or more samples")
        else:
            print("⚠️  ACCEPTABLE: May work but not ideal")
    else:
        print(f"Power ratio: {power_ratio:.4f}")
        print(f"Cohen's d (power): {cohens_d_power:.4f}")
        print()
        print("💡 Tip: Run with --spectral flag for full analysis")
    
    print("="*60)

if __name__ == "__main__":
    dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    
    # Check for --spectral flag
    check_spectral = '--spectral' in sys.argv or '-s' in sys.argv
    
    try:
        analyze_power_and_spectral(dataset_path, check_spectral)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
