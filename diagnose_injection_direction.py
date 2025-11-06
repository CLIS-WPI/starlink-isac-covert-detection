#!/usr/bin/env python3
"""
Diagnostic: Check if injection is actually INCREASING magnitude in covert band
"""
import numpy as np
import pickle
from pathlib import Path

# Load dataset
dataset_path = Path("dataset/dataset_samples500_sats12.pkl")
print(f"Loading dataset from: {dataset_path}")
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

# Extract data based on labels (0=benign, 1=attack)
labels = data['labels']
rx_grids = data['rx_grids']  # (1000, 1, 10, 64)
tx_grids = data['tx_grids']  # (1000, 1, 1, 1, 10, 64)

# Split by label
benign_mask = (labels == 0)
attack_mask = (labels == 1)

benign_rx = rx_grids[benign_mask]  # (500, 1, 10, 64)
attack_rx = rx_grids[attack_mask]
benign_tx = tx_grids[benign_mask]  # (500, 1, 1, 1, 10, 64)
attack_tx = tx_grids[attack_mask]

print(f"Loaded {len(benign_rx)} benign and {len(attack_rx)} attack samples")

# Remove extra dimensions
benign_rx = benign_rx.squeeze(1)  # (500, 10, 64)
attack_rx = attack_rx.squeeze(1)
benign_tx = benign_tx.squeeze()  # (500, 10, 64)
attack_tx = attack_tx.squeeze()

print("=" * 80)
print("INJECTION DIRECTION DIAGNOSTIC")
print("=" * 80)

# Test 1: Are tx_grids different between benign and attack?
print("\n1️⃣ TX GRIDS COMPARISON (before channel):")
benign_tx_power = np.mean(np.abs(benign_tx) ** 2)
attack_tx_power = np.mean(np.abs(attack_tx) ** 2)
print(f"Benign TX power: {benign_tx_power:.6f}")
print(f"Attack TX power: {attack_tx_power:.6f}")
print(f"Ratio (attack/benign): {attack_tx_power / benign_tx_power:.4f}")

if np.allclose(benign_tx, attack_tx):
    print("⚠️  WARNING: TX grids are IDENTICAL! Injection may not be happening in TX.")
else:
    print("✅ TX grids are DIFFERENT (injection applied)")

# Test 2: Are rx_grids different between benign and attack?
print("\n2️⃣ RX GRIDS COMPARISON (after channel):")
benign_rx_power = np.mean(np.abs(benign_rx) ** 2)
attack_rx_power = np.mean(np.abs(attack_rx) ** 2)
print(f"Benign RX power: {benign_rx_power:.6f}")
print(f"Attack RX power: {attack_rx_power:.6f}")
print(f"Ratio (attack/benign): {attack_rx_power / benign_rx_power:.4f}")

if np.allclose(benign_rx, attack_rx):
    print("❌ CRITICAL: RX grids are IDENTICAL! Injection is NOT working!")
else:
    print("✅ RX grids are DIFFERENT")

# Test 3: Covert band magnitude comparison (RX)
print("\n3️⃣ COVERT BAND (subcarriers 0-15) in RX:")
covert_band = slice(0, 16)
benign_covert = np.abs(benign_rx[:, :, covert_band])  # (500, 10, 16)
attack_covert = np.abs(attack_rx[:, :, covert_band])

benign_covert_mean = np.mean(benign_covert)
attack_covert_mean = np.mean(attack_covert)
diff_pct = 100 * (attack_covert_mean - benign_covert_mean) / benign_covert_mean

print(f"Benign covert magnitude: {benign_covert_mean:.6f}")
print(f"Attack covert magnitude: {attack_covert_mean:.6f}")
print(f"Difference: {diff_pct:+.2f}%")

if attack_covert_mean > benign_covert_mean:
    print("✅ Attack > Benign (as expected)")
else:
    print("❌ PROBLEM: Attack < Benign (opposite of expected!)")

# Test 4: Symbol-specific analysis
print("\n4️⃣ SYMBOL-SPECIFIC ANALYSIS (symbols 1,3,5,7):")
covert_symbols = [1, 3, 5, 7]
for sym_idx in covert_symbols:
    benign_sym = np.abs(benign_rx[:, sym_idx, covert_band])
    attack_sym = np.abs(attack_rx[:, sym_idx, covert_band])
    
    benign_mean = np.mean(benign_sym)
    attack_mean = np.mean(attack_sym)
    diff = 100 * (attack_mean - benign_mean) / benign_mean
    
    status = "✅" if attack_mean > benign_mean else "❌"
    print(f"  Symbol {sym_idx}: benign={benign_mean:.6f}, attack={attack_mean:.6f}, diff={diff:+.2f}% {status}")

# Test 5: Sample-level variance
print("\n5️⃣ SAMPLE-LEVEL VARIANCE:")
sample_diffs = []
for i in range(min(10, len(benign_rx))):
    benign_sample = np.mean(np.abs(benign_rx[i, :, covert_band]))
    attack_sample = np.mean(np.abs(attack_rx[i, :, covert_band]))
    diff = 100 * (attack_sample - benign_sample) / benign_sample
    sample_diffs.append(diff)
    print(f"  Sample {i}: benign={benign_sample:.6f}, attack={attack_sample:.6f}, diff={diff:+.2f}%")

print(f"\nMean of sample differences: {np.mean(sample_diffs):+.2f}%")
print(f"Std of sample differences: {np.std(sample_diffs):.2f}%")

# Test 6: Check if power preservation is causing the issue
print("\n6️⃣ POWER PRESERVATION CHECK:")
benign_total_power = np.mean(np.abs(benign_rx) ** 2, axis=(1,2))  # (500,)
attack_total_power = np.mean(np.abs(attack_rx) ** 2, axis=(1,2))

print(f"Benign total power: {np.mean(benign_total_power):.6f} ± {np.std(benign_total_power):.6f}")
print(f"Attack total power: {np.mean(attack_total_power):.6f} ± {np.std(attack_total_power):.6f}")

if np.mean(attack_total_power) < np.mean(benign_total_power) * 0.98:
    print("⚠️  WARNING: Attack samples have LOWER total power!")
    print("   Power preservation may be OVER-reducing attack samples")

# Test 7: TX vs RX comparison for attack samples
print("\n7️⃣ TX vs RX COMPARISON (attack samples only):")
attack_tx_covert = np.abs(attack_tx[:, :, covert_band])
attack_rx_covert = np.abs(attack_rx[:, :, covert_band])

print(f"Attack TX covert: {np.mean(attack_tx_covert):.6f}")
print(f"Attack RX covert: {np.mean(attack_rx_covert):.6f}")
print(f"Channel attenuation: {100 * (1 - np.mean(attack_rx_covert) / np.mean(attack_tx_covert)):.1f}%")

# Test 8: Check if injection is in the right symbols
print("\n8️⃣ INJECTION SYMBOL CHECK:")
benign_symbols = [0, 2, 4, 6, 8, 9]  # Non-covert symbols
for sym_idx in benign_symbols:
    benign_sym = np.mean(np.abs(benign_rx[:, sym_idx, covert_band]))
    attack_sym = np.mean(np.abs(attack_rx[:, sym_idx, covert_band]))
    diff = 100 * (attack_sym - benign_sym) / benign_sym
    
    if abs(diff) > 2.0:
        print(f"  ⚠️  Symbol {sym_idx} (should be clean): diff={diff:+.2f}% (leakage!)")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
if attack_covert_mean < benign_covert_mean:
    print("🔴 CRITICAL ISSUE: Injection is DECREASING magnitude in covert band!")
    print("   Possible causes:")
    print("   1. Power preservation is over-correcting attack samples")
    print("   2. Injection is being applied with NEGATIVE amplitude")
    print("   3. Injection is happening in wrong location/time")
else:
    print("✅ Injection direction is correct (attack > benign)")
    print("   The problem is likely variance or SNR-related")
