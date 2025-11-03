#!/usr/bin/env python3
"""
Test script to verify power normalization is working correctly
"""

import pickle
import numpy as np

def analyze_power_normalization(dataset_path):
    """Analyze power distribution before/after normalization"""
    print("=" * 60)
    print("POWER NORMALIZATION TEST")
    print("=" * 60)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    iq_samples = dataset['iq_samples']
    labels = dataset['labels']
    
    # Separate benign and attack
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    attack_indices = [i for i, label in enumerate(labels) if label == 1]
    
    print(f"\n📊 Dataset: {len(labels)} samples")
    print(f"  Benign: {len(benign_indices)}")
    print(f"  Attack: {len(attack_indices)}")
    
    # Calculate powers
    benign_powers = [np.mean(np.abs(iq_samples[i])**2) for i in benign_indices]
    attack_powers = [np.mean(np.abs(iq_samples[i])**2) for i in attack_indices]
    
    benign_mean = np.mean(benign_powers)
    benign_std = np.std(benign_powers)
    attack_mean = np.mean(attack_powers)
    attack_std = np.std(attack_powers)
    
    power_ratio = attack_mean / benign_mean if benign_mean > 0 else 0
    
    # Cohen's d
    pooled_std = np.sqrt((benign_std**2 + attack_std**2) / 2)
    cohens_d = (attack_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
    
    print(f"\n📈 POWER STATISTICS:")
    print(f"  Benign: {benign_mean:.6e} ± {benign_std:.6e}")
    print(f"  Attack: {attack_mean:.6e} ± {attack_std:.6e}")
    print(f"\n🎯 KEY METRICS:")
    print(f"  Power Ratio (attack/benign): {power_ratio:.4f}")
    print(f"  Cohen's d (power domain):    {cohens_d:.4f}")
    
    # Verdict
    print(f"\n{'=' * 60}")
    print("VERDICT:")
    if 0.95 <= power_ratio <= 1.05:
        print("✅ EXCELLENT: Power ratio ≈ 1.0 (truly covert!)")
    elif 0.90 <= power_ratio <= 1.10:
        print("✅ GOOD: Power ratio close to 1.0 (mostly covert)")
    elif 0.80 <= power_ratio <= 1.20:
        print("⚠️  WARNING: Power ratio slightly off (detectable)")
    else:
        print("❌ FAIL: Power ratio too far from 1.0 (NOT covert!)")
    
    if abs(cohens_d) < 0.1:
        print("✅ Cohen's d < 0.1 (excellent stealth in power domain)")
    elif abs(cohens_d) < 0.2:
        print("✅ Cohen's d < 0.2 (good stealth in power domain)")
    else:
        print("⚠️  Cohen's d ≥ 0.2 (detectable in power domain)")
    
    print(f"{'=' * 60}\n")
    
    # Show first 10 samples
    print("📋 First 10 samples of each class:")
    print("-" * 60)
    print(f"{'Index':<8} {'Label':<10} {'Power':<20}")
    print("-" * 60)
    for i in benign_indices[:10]:
        power = np.mean(np.abs(iq_samples[i])**2)
        print(f"{i:<8} {'BENIGN':<10} {power:.6e}")
    print()
    for i in attack_indices[:10]:
        power = np.mean(np.abs(iq_samples[i])**2)
        print(f"{i:<8} {'ATTACK':<10} {power:.6e}")
    print("-" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Default path
        dataset_path = "dataset/dataset_samples1500_sats12.pkl"
    
    try:
        analyze_power_normalization(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Usage: python3 test_power_normalization.py [dataset_path]")
        sys.exit(1)
