#!/usr/bin/env python3
"""
Debug localization failures by checking GCC-PHAT correlation quality.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
with open('dataset/dataset_samples1500_sats12.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Pick first attack sample
sample_idx = 1500
print(f"\n=== Analyzing sample {sample_idx} ===")

sats = dataset['satellite_receptions'][sample_idx]
print(f"Number of satellites: {len(sats)}")

# Get reference signal
ref_sig = dataset['tx_time_padded'][sample_idx]
print(f"Reference signal length: {len(ref_sig)}")
print(f"Reference power: {np.mean(np.abs(ref_sig)**2):.6e}")

# Get ground truth
emit_loc = dataset['emitter_locations'][sample_idx]
print(f"Emitter location: {emit_loc}")

# Check correlation for first 3 satellites
print("\n=== GCC-PHAT Correlation Quality ===")

from core.localization import _estimate_toa

Fs = dataset['sampling_rate']
c0 = 3e8

for i in range(min(3, len(sats))):
    sat = sats[i]
    rx_sig = sat.get('rx_time_b_full', sat['rx_time_padded'])
    
    print(f"\nSatellite {i}:")
    print(f"  Position: {sat['position']}")
    print(f"  RX signal length: {len(rx_sig)}")
    print(f"  RX power: {np.mean(np.abs(rx_sig)**2):.6e}")
    
    # Estimate TOA
    try:
        dt, curv, score = _estimate_toa(rx_sig, ref_sig, Fs)
        
        # Compute expected TOA from geometry
        if emit_loc is not None:
            # Distance from emitter to satellite
            dist = np.linalg.norm(np.array(sat['position']) - emit_loc)
        else:
            # Distance from default ground point
            dist = np.linalg.norm(np.array(sat['position']) - np.array([50e3, 50e3, 0]))
        
        expected_delay_s = dist / c0
        
        print(f"  Estimated TOA: {dt*1e6:.2f} μs")
        print(f"  Expected TOA: {expected_delay_s*1e6:.2f} μs")
        print(f"  Error: {abs(dt - expected_delay_s)*1e6:.2f} μs = {abs(dt - expected_delay_s)*c0/1e3:.1f} km")
        print(f"  Correlation score: {score:.6f}")
        print(f"  Curvature: {curv:.6f}")
        
        if abs(dt - expected_delay_s) > 1e-3:  # > 1 ms = 300 km error
            print(f"  ⚠️ LARGE ERROR: GCC-PHAT correlation is failing!")
            
            # Check if signals are too different
            # Cross-correlation magnitude
            L = min(len(rx_sig), len(ref_sig))
            xcorr_max = np.max(np.abs(np.correlate(rx_sig[:L], ref_sig[:L], mode='full')))
            autocorr_rx = np.max(np.abs(np.correlate(rx_sig[:L], rx_sig[:L], mode='full')))
            autocorr_ref = np.max(np.abs(np.correlate(ref_sig[:L], ref_sig[:L], mode='full')))
            norm_xcorr = xcorr_max / np.sqrt(autocorr_rx * autocorr_ref + 1e-12)
            print(f"  Normalized cross-corr: {norm_xcorr:.4f} (should be >0.5)")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n=== DIAGNOSIS ===")
print("If TOA errors are >100 km, likely causes:")
print("1. tx_time_padded is NOT the clean transmitted signal")
print("2. rx_time_b_full has too much noise (SNR too low)")
print("3. Satellite geometry makes TDOA ill-conditioned")
print("\nRecommendation: Regenerate dataset with higher SNR (Eb/N0 > 15 dB)")
