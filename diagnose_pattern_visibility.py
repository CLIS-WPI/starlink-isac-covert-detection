#!/usr/bin/env python3
"""
Diagnostic: Check if covert pattern is visible in rx_grids
"""
import pickle
import numpy as np

# Load dataset
print("Loading dataset...")
d = pickle.load(open('dataset/dataset_samples500_sats12.pkl', 'rb'))
rx_grids = d['rx_grids']
labels = d['labels']

print(f"rx_grids shape: {rx_grids.shape}")
print(f"Labels: {np.sum(labels==0)} benign, {np.sum(labels==1)} attack")

# Separate benign and attack
benign_grids = rx_grids[labels == 0]
attack_grids = rx_grids[labels == 1]

# Squeeze to remove singleton dimension: (200, 1, 10, 64) -> (200, 10, 64)
benign_grids = np.squeeze(benign_grids, axis=1)
attack_grids = np.squeeze(attack_grids, axis=1)

print(f"Benign grids shape: {benign_grids.shape}")
print(f"Attack grids shape: {attack_grids.shape}")

# Compute average spectrum per class (average over samples and OFDM symbols)
benign_spectrum = np.mean(np.abs(benign_grids), axis=(0, 1))  # (64,)
attack_spectrum = np.mean(np.abs(attack_grids), axis=(0, 1))  # (64,)

# Check first 16 subcarriers (where covert pattern should be)
print("\n" + "="*70)
print("SPECTRAL PATTERN ANALYSIS")
print("="*70)
print("\nAverage magnitude per subcarrier (first 20):")
print(f"{'Subcarrier':<12} {'Benign':<15} {'Attack':<15} {'Diff %':<10}")
print("-"*70)
for i in range(20):
    diff_pct = (attack_spectrum[i] - benign_spectrum[i]) / benign_spectrum[i] * 100
    print(f"{i:<12} {benign_spectrum[i]:<15.6f} {attack_spectrum[i]:<15.6f} {diff_pct:>8.2f}%")

# Overall statistics
print("\n" + "="*70)
print("COVERT BAND (subcarriers 0-15):")
benign_covert = np.mean(benign_spectrum[:16])
attack_covert = np.mean(attack_spectrum[:16])
diff_covert = (attack_covert - benign_covert) / benign_covert * 100
print(f"  Benign avg: {benign_covert:.6f}")
print(f"  Attack avg: {attack_covert:.6f}")
print(f"  Difference: {diff_covert:.2f}%")

print("\nNORMAL BAND (subcarriers 16-63):")
benign_normal = np.mean(benign_spectrum[16:])
attack_normal = np.mean(attack_spectrum[16:])
diff_normal = (attack_normal - benign_normal) / benign_normal * 100
print(f"  Benign avg: {benign_normal:.6f}")
print(f"  Attack avg: {attack_normal:.6f}")
print(f"  Difference: {diff_normal:.2f}%")

# Check individual samples
print("\n" + "="*70)
print("SAMPLE-LEVEL ANALYSIS (first 5 of each class):")
print("="*70)
for i in range(5):
    benign_sample = benign_grids[i, :, :]  # (10, 64)
    attack_sample = attack_grids[i, :, :]  # (10, 64)
    
    # Average over OFDM symbols
    benign_avg = np.mean(np.abs(benign_sample), axis=0)
    attack_avg = np.mean(np.abs(attack_sample), axis=0)
    
    # Covert band difference
    benign_covert_power = np.mean(benign_avg[:16])
    attack_covert_power = np.mean(attack_avg[:16])
    diff = (attack_covert_power - benign_covert_power) / benign_covert_power * 100
    
    print(f"Sample {i}: Benign covert band = {benign_covert_power:.6f}, "
          f"Attack covert band = {attack_covert_power:.6f}, Diff = {diff:>6.2f}%")

print("\n" + "="*70)
if abs(diff_covert) > 5.0:
    print("✅ PATTERN VISIBLE: Covert band shows >5% difference")
    print("   → CNN should be able to detect this")
else:
    print("❌ PATTERN WEAK: Covert band shows <5% difference")
    print("   → Pattern may be too subtle for CNN to detect")
    print("   → Consider increasing COVERT_AMP or checking injection code")
print("="*70)
