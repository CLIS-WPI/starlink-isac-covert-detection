#!/usr/bin/env python3
"""
Validate Path-B Fix: Check if dataset has been fixed properly
Tests:
  1. Cross-correlation quality (should be >0.5 after fix)
  2. Dataset structure validation
  3. Power ratio sanity check
"""

import pickle
import numpy as np
import sys

print("=" * 70)
print("DATASET VALIDATION - PATH-B FIX")
print("=" * 70)

# Load dataset
print("\n[1/3] Loading dataset...")
try:
    with open('dataset/dataset_samples1500_sats12.pkl', 'rb') as f:
        dataset = pickle.load(f)
    print("✓ Dataset loaded successfully")
except FileNotFoundError:
    print("❌ ERROR: Dataset not found!")
    print("   Please run: python3 generate_dataset_parallel.py")
    sys.exit(1)

# Basic validation
print("\n[2/3] Basic Dataset Validation...")
total_samples = len(dataset['labels'])
attack_samples = sum(dataset['labels'])
benign_samples = total_samples - attack_samples

print(f"  Total samples: {total_samples}")
print(f"  Benign: {benign_samples}")
print(f"  Attack: {attack_samples}")

if total_samples != 3000:
    print(f"  ⚠️ WARNING: Expected 3000 samples, got {total_samples}")

# Test sample structure
sample_idx = 1500  # First attack sample
sats = dataset['satellite_receptions'][sample_idx]
tx_ref = dataset['tx_time_padded'][sample_idx]

print(f"\n  Sample {sample_idx} (attack):")
print(f"    Satellites: {len(sats)}")
print(f"    TX reference length: {len(tx_ref)}")
print(f"    RX Path-B length (sat 0): {len(sats[0]['rx_time_b_full'])}")

# Check power ratio
tx_pow = np.mean(np.abs(tx_ref)**2)
rx_pow = np.mean(np.abs(sats[0]['rx_time_b_full'])**2)
power_ratio = rx_pow / tx_pow

print(f"    TX power: {tx_pow:.6e}")
print(f"    RX power: {rx_pow:.6e}")
print(f"    Power ratio (RX/TX): {power_ratio:.2f}")

if power_ratio < 0.5 or power_ratio > 5.0:
    print(f"    ⚠️ WARNING: Unusual power ratio (expected 1-2)")
else:
    print(f"    ✓ Power ratio OK")

# Critical test: Cross-correlation quality
print("\n[3/3] Cross-Correlation Quality Test (CRITICAL)...")
print("  This is the key test to verify Path-B fix!")
print()

from core.localization import _estimate_toa

Fs = dataset['sampling_rate']
c0 = 3e8

# Test first 3 satellites
good_count = 0
bad_count = 0
correlation_scores = []

for i in range(min(3, len(sats))):
    sat = sats[i]
    rx_sig = sat.get('rx_time_b_full', sat['rx_time_padded'])
    
    print(f"  Satellite {i}:")
    
    # Estimate TOA
    try:
        dt, curv, score = _estimate_toa(rx_sig, tx_ref, Fs)
        
        # Compute expected TOA from geometry
        emit_loc = dataset['emitter_locations'][sample_idx]
        if emit_loc is not None:
            dist = np.linalg.norm(np.array(sat['position']) - emit_loc)
        else:
            dist = np.linalg.norm(np.array(sat['position']) - np.array([50e3, 50e3, 0]))
        
        expected_delay_s = dist / c0
        toa_error = abs(dt - expected_delay_s)
        
        # Normalized cross-correlation
        L = min(len(rx_sig), len(tx_ref))
        xcorr = np.correlate(rx_sig[:L], tx_ref[:L], mode='full')
        xcorr_max = np.max(np.abs(xcorr))
        autocorr_rx = np.max(np.abs(np.correlate(rx_sig[:L], rx_sig[:L], mode='full')))
        autocorr_ref = np.max(np.abs(np.correlate(tx_ref[:L], tx_ref[:L], mode='full')))
        norm_xcorr = xcorr_max / np.sqrt(autocorr_rx * autocorr_ref + 1e-12)
        
        correlation_scores.append(norm_xcorr)
        
        print(f"    TOA estimate: {dt*1e6:.2f} μs")
        print(f"    TOA expected: {expected_delay_s*1e6:.2f} μs")
        print(f"    TOA error: {toa_error*1e6:.2f} μs = {toa_error*c0/1e3:.1f} km")
        print(f"    Normalized cross-corr: {norm_xcorr:.4f}")
        
        # Evaluation
        if norm_xcorr >= 0.5 and toa_error < 100e3:  # <100 km
            print(f"    ✅ GOOD: High correlation, low error")
            good_count += 1
        elif norm_xcorr >= 0.3:
            print(f"    ⚠️  FAIR: Medium correlation")
        else:
            print(f"    ❌ POOR: Low correlation - Path-B fix may have failed!")
            bad_count += 1
        
        print()
        
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        bad_count += 1
        print()

# Summary
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

avg_corr = np.mean(correlation_scores) if correlation_scores else 0

print(f"\nCross-Correlation Statistics:")
print(f"  Average: {avg_corr:.4f}")
print(f"  Min: {min(correlation_scores):.4f}" if correlation_scores else "  Min: N/A")
print(f"  Max: {max(correlation_scores):.4f}" if correlation_scores else "  Max: N/A")
print(f"  Good satellites: {good_count}/3")
print(f"  Poor satellites: {bad_count}/3")

print("\nExpected values:")
print("  BEFORE fix: correlation ~0.16, TOA error ~700-1400 km")
print("  AFTER fix:  correlation >0.50, TOA error <100 km")

# Final verdict
print("\n" + "=" * 70)
if avg_corr >= 0.5:
    print("✅ VALIDATION PASSED!")
    print("   Path-B fix is working correctly!")
    print("   You can proceed to run: python3 main.py")
    print("=" * 70)
    sys.exit(0)
elif avg_corr >= 0.3:
    print("⚠️  VALIDATION PARTIAL")
    print("   Path-B fix shows improvement but not optimal")
    print("   Expected: Cross-correlation >0.5")
    print(f"   Actual:   Cross-correlation {avg_corr:.4f}")
    print("   Consider investigating further before running main.py")
    print("=" * 70)
    sys.exit(1)
else:
    print("❌ VALIDATION FAILED!")
    print("   Path-B fix did NOT work as expected")
    print(f"   Cross-correlation is still low: {avg_corr:.4f}")
    print("   DO NOT run main.py - it will waste time")
    print("\n   Possible issues:")
    print("   1. Dataset was not regenerated after fix")
    print("   2. Fix was not applied correctly")
    print("   3. There is another underlying issue")
    print("=" * 70)
    sys.exit(1)
