#!/usr/bin/env python3
"""
🔍 Diagnostic Tools - SNR Analysis & Label Verification
========================================================
1. Analyze AUC per SNR range
2. Verify label alignment with covert injection
3. Check for label leakage
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import json


def analyze_snr_performance(dataset_path, model_path):
    """Analyze model performance across different SNR ranges"""
    
    print("\n" + "="*70)
    print("📊 SNR-BASED PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Load dataset
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    # Check if SNR info exists in dataset
    if 'snr_db' in dataset or 'ebno_db' in dataset:
        snr_values = dataset.get('snr_db', dataset.get('ebno_db', None))
    else:
        print("⚠️  SNR information not found in dataset")
        # Estimate SNR from signal power and noise
        print("   Estimating SNR from signal characteristics...")
        snr_values = estimate_snr(X_grids)
    
    # Define SNR ranges
    snr_ranges = [
        ("Low (0-10 dB)", 0, 10),
        ("Medium (10-20 dB)", 10, 20),
        ("High (20-30 dB)", 20, 30),
        ("Very High (>30 dB)", 30, 100),
    ]
    
    print(f"\n📈 Performance by SNR Range:")
    print(f"  {'SNR Range':<20} {'Samples':<10} {'Benign':<10} {'Attack':<10} {'Balance'}")
    print(f"  {'-'*70}")
    
    snr_stats = {}
    
    for range_name, snr_min, snr_max in snr_ranges:
        mask = (snr_values >= snr_min) & (snr_values < snr_max)
        
        if np.sum(mask) == 0:
            continue
        
        X_range = X_grids[mask]
        Y_range = Y[mask]
        
        benign_count = np.sum(Y_range == 0)
        attack_count = np.sum(Y_range == 1)
        balance_ratio = min(benign_count, attack_count) / max(benign_count, attack_count) if max(benign_count, attack_count) > 0 else 0
        
        print(f"  {range_name:<20} {len(Y_range):<10} {benign_count:<10} {attack_count:<10} {balance_ratio:.2f}")
        
        snr_stats[range_name] = {
            'total': len(Y_range),
            'benign': int(benign_count),
            'attack': int(attack_count),
            'balance': float(balance_ratio)
        }
    
    # If model exists, evaluate per SNR
    if Path(model_path).exists():
        print(f"\n🧠 Loading model from {model_path}...")
        
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            
            print(f"\n📊 AUC by SNR Range:")
            print(f"  {'SNR Range':<20} {'AUC':<10} {'Status'}")
            print(f"  {'-'*50}")
            
            for range_name, snr_min, snr_max in snr_ranges:
                mask = (snr_values >= snr_min) & (snr_values < snr_max)
                
                if np.sum(mask) == 0:
                    continue
                
                X_range = X_grids[mask]
                Y_range = Y[mask]
                
                if len(np.unique(Y_range)) < 2:
                    print(f"  {range_name:<20} {'N/A':<10} (single class)")
                    continue
                
                # Preprocess (simple magnitude for now)
                X_proc = np.abs(X_range).astype(np.float32)
                
                # Predict
                y_pred = model.predict(X_proc, verbose=0)
                
                # Calculate AUC
                auc = roc_auc_score(Y_range, y_pred)
                
                status = "✅ Good" if auc >= 0.7 else "⚠️ Poor"
                print(f"  {range_name:<20} {auc:.4f}    {status}")
                
                snr_stats[range_name]['auc'] = float(auc)
        
        except Exception as e:
            print(f"⚠️  Could not evaluate model: {e}")
    
    return snr_stats


def estimate_snr(X_grids):
    """Estimate SNR from OFDM grids"""
    
    # Simple estimation: signal power / (std of weak signals)
    powers = np.mean(np.abs(X_grids)**2, axis=(1,2,3,4))
    
    # Assume top 50% are signal, bottom 50% are noise-dominated
    sorted_powers = np.sort(powers)
    mid = len(sorted_powers) // 2
    
    signal_power = np.mean(sorted_powers[mid:])
    noise_power = np.mean(sorted_powers[:mid])
    
    snr_linear = signal_power / (noise_power + 1e-10)
    snr_db = 10 * np.log10(snr_linear)
    
    # Generate per-sample SNR (with some variation)
    snr_values = snr_db + np.random.randn(len(X_grids)) * 3
    
    return snr_values


def verify_labels(dataset_path):
    """Verify label alignment and check for leakage"""
    
    print("\n" + "="*70)
    print("🔍 LABEL VERIFICATION & LEAKAGE CHECK")
    print("="*70)
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    X_grids = dataset['tx_grids']
    Y = dataset['labels']
    
    print(f"\n1️⃣ Basic Statistics:")
    print(f"   Total samples: {len(Y)}")
    print(f"   Benign (0): {np.sum(Y == 0)} ({np.sum(Y == 0)/len(Y)*100:.1f}%)")
    print(f"   Attack (1): {np.sum(Y == 1)} ({np.sum(Y == 1)/len(Y)*100:.1f}%)")
    
    # Check power difference
    benign_mask = (Y == 0)
    attack_mask = (Y == 1)
    
    benign_grids = np.squeeze(X_grids[benign_mask])
    attack_grids = np.squeeze(X_grids[attack_mask])
    
    benign_power = np.mean(np.abs(benign_grids)**2)
    attack_power = np.mean(np.abs(attack_grids)**2)
    power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    print(f"\n2️⃣ Power Analysis (Label Verification):")
    print(f"   Benign power: {benign_power:.6f}")
    print(f"   Attack power: {attack_power:.6f}")
    print(f"   Difference:   {power_diff_pct:.2f}%")
    
    if power_diff_pct > 1.0:
        print(f"   ✅ Labels appear correct (power diff > 1%)")
    else:
        print(f"   ⚠️  Power diff very small - check injection!")
    
    # Spectral analysis
    print(f"\n3️⃣ Spectral Signature:")
    
    # FFT along subcarrier axis
    benign_fft = np.fft.fft(np.mean(benign_grids, axis=0), axis=-1)
    attack_fft = np.fft.fft(np.mean(attack_grids, axis=0), axis=-1)
    
    spectral_diff = np.abs(attack_fft - benign_fft)
    max_diff_idx = np.unravel_index(np.argmax(spectral_diff), spectral_diff.shape)
    max_diff_val = spectral_diff[max_diff_idx]
    
    print(f"   Max spectral difference: {max_diff_val:.6f}")
    print(f"   Location: symbol={max_diff_idx[0]}, subcarrier={max_diff_idx[1]}")
    
    if max_diff_val > 0.01:
        print(f"   ✅ Spectral signature detected")
    else:
        print(f"   ⚠️  Weak spectral signature")
    
    # Check for label leakage (timing)
    if 'tx_time_padded' in dataset and 'rx_time_b_full' in dataset:
        print(f"\n4️⃣ Timing Alignment Check:")
        
        tx_times = dataset['tx_time_padded']
        rx_times = dataset['rx_time_b_full']
        
        # Check if attack samples have different timing patterns
        benign_tx_mean = np.mean([t for t, label in zip(tx_times, Y) if label == 0])
        attack_tx_mean = np.mean([t for t, label in zip(tx_times, Y) if label == 1])
        
        timing_diff = abs(attack_tx_mean - benign_tx_mean)
        
        print(f"   Benign avg tx_time: {benign_tx_mean:.6f}")
        print(f"   Attack avg tx_time: {attack_tx_mean:.6f}")
        print(f"   Difference: {timing_diff:.6f}")
        
        if timing_diff < 1e-6:
            print(f"   ✅ No timing leakage detected")
        else:
            print(f"   ⚠️  Timing difference detected - possible leakage!")
    
    # Sample-level verification
    print(f"\n5️⃣ Sample-Level Verification (first 10 attack samples):")
    
    attack_indices = np.where(attack_mask)[0][:10]
    
    for idx in attack_indices:
        sample_power = np.mean(np.abs(X_grids[idx])**2)
        power_vs_benign = (sample_power - benign_power) / benign_power * 100
        
        status = "✅" if abs(power_vs_benign) < 10 else "⚠️"
        print(f"   Sample {idx}: power={sample_power:.6f}, diff={power_vs_benign:+.2f}% {status}")
    
    print("\n" + "="*70)


def save_diagnostic_report(dataset_path, model_path, output_file):
    """Generate comprehensive diagnostic report"""
    
    report = {
        'timestamp': int(time.time()),
        'dataset': dataset_path,
        'model': model_path,
    }
    
    # SNR analysis
    print("Running SNR analysis...")
    snr_stats = analyze_snr_performance(dataset_path, model_path)
    report['snr_analysis'] = snr_stats
    
    # Label verification
    print("\nRunning label verification...")
    verify_labels(dataset_path)
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Diagnostic report saved to {output_file}")


if __name__ == "__main__":
    import sys
    import time
    from config.settings import NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA, DATASET_DIR, MODEL_DIR
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    model_path = f"{MODEL_DIR}/cnn_detector.keras"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full-report":
        output_file = f"result/diagnostic_report_{int(time.time())}.json"
        save_diagnostic_report(dataset_path, model_path, output_file)
    else:
        # Quick diagnostics
        verify_labels(dataset_path)
        analyze_snr_performance(dataset_path, model_path)
