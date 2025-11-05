#!/usr/bin/env python3
"""
🔍 Quick Power Analysis
=======================
Analyze power difference in dataset before training CNN
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_dataset(dataset_path):
    """Analyze power difference and injection pattern"""
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("\n" + "="*60)
    print("🔍 DATASET POWER ANALYSIS")
    print("="*60)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    benign_mask = (Y == 0)
    attack_mask = (Y == 1)
    
    benign_grids = np.squeeze(X_grids[benign_mask])
    attack_grids = np.squeeze(X_grids[attack_mask])
    
    # Power analysis
    benign_power = np.mean(np.abs(benign_grids) ** 2)
    attack_power = np.mean(np.abs(attack_grids) ** 2)
    power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    print(f"\n📊 Power Statistics:")
    print(f"  Benign power: {benign_power:.6f}")
    print(f"  Attack power: {attack_power:.6f}")
    print(f"  Difference:   {power_diff_pct:.2f}%")
    
    # Expected AUC based on power difference
    if power_diff_pct < 1.0:
        expected_auc = "0.50-0.60"
        status = "❌ TOO WEAK - CNN unlikely to learn"
        recommendation = "Increase COVERT_AMP to 1.4-1.6"
    elif power_diff_pct < 2.0:
        expected_auc = "0.60-0.70"
        status = "⚠️ WEAK - May need more epochs"
        recommendation = "Consider increasing COVERT_AMP to 1.4"
    elif power_diff_pct < 4.0:
        expected_auc = "0.70-0.85"
        status = "✅ GOOD - CNN should learn"
        recommendation = "Proceed with training"
    elif power_diff_pct < 8.0:
        expected_auc = "0.85-0.93"
        status = "✅ STRONG - Excellent for training"
        recommendation = "After success, try reducing COVERT_AMP"
    else:
        expected_auc = "0.90-0.98"
        status = "⚠️ TOO STRONG - Not realistic covert"
        recommendation = "Reduce COVERT_AMP to 1.2-1.4"
    
    print(f"\n🎯 Expected Performance:")
    print(f"  Status:        {status}")
    print(f"  Expected AUC:  {expected_auc}")
    print(f"  Recommendation: {recommendation}")
    
    # Spectral variation analysis
    print(f"\n📈 Sample Variation:")
    
    # Check variation in first 10 attack samples
    n_check = min(10, len(attack_grids))
    attack_sample = attack_grids[:n_check]
    
    # Calculate variance across frequency domain
    freq_variance = np.var(np.abs(attack_sample), axis=0).mean()
    print(f"  Spectral variance: {freq_variance:.6f}")
    
    if freq_variance > 0.001:
        print(f"  ✅ Good variation - CNN can learn diverse patterns")
    else:
        print(f"  ⚠️ Low variation - Check RANDOMIZE_* settings")
    
    print("\n" + "="*60)
    print("💡 TIP: Run CNN training if power diff > 2%")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    from config.settings import NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA, DATASET_DIR
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    analyze_dataset(dataset_path)
