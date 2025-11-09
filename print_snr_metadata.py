#!/usr/bin/env python3
"""
📊 Print Detailed SNR Metadata
===============================
Prints detailed metadata for SNR analysis.
"""

import pickle
import numpy as np
import sys

def print_metadata(dataset_path, num_samples=100):
    """Print detailed metadata for SNR analysis."""
    
    print("="*70)
    print("📊 Detailed SNR Metadata Analysis")
    print("="*70)
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    
    print(f"\n📊 Analyzing {min(num_samples, len(meta_list))} samples...\n")
    
    # Print header
    print(f"{'Sample':<8} {'SNR_in':<8} {'SNR_raw':<10} {'SNR_eq':<10} {'ΔSNR':<8} "
          f"{'α':<12} {'α/|H|²':<10} {'|H|²':<12} {'blend':<8} {'pres':<8}")
    print("-"*110)
    
    for i in range(min(num_samples, len(meta_list))):
        meta = meta_list[i]
        if isinstance(meta, tuple):
            _, meta = meta
        
        snr_input = meta.get('snr_dl', meta.get('snr_db', 0))
        snr_raw = meta.get('eq_snr_raw_db', 0)
        snr_eq = meta.get('eq_snr_db', 0)
        snr_imp = meta.get('eq_snr_improvement_db', 0)
        alpha = meta.get('eq_alpha', meta.get('alpha_used', 0))
        alpha_ratio = meta.get('alpha_ratio', meta.get('eq_alpha_ratio', 0))
        h_power = meta.get('h_power_mean', meta.get('csi_P_H', 0))
        blend = meta.get('eq_blend_factor', 1.0)
        pres = meta.get('eq_pattern_preservation', 0)
        
        print(f"{i:<8} {snr_input:<8.1f} {snr_raw:<10.2f} {snr_eq:<10.2f} {snr_imp:<8.2f} "
              f"{alpha:<12.6e} {alpha_ratio:<10.3f} {h_power:<12.6e} {blend:<8.2f} {pres:<8.3f}")
    
    # Statistics
    snr_improvements = []
    alpha_ratios = []
    preservations = []
    
    for meta in meta_list[:num_samples]:
        if isinstance(meta, tuple):
            _, meta = meta
        
        if 'eq_snr_improvement_db' in meta:
            snr_improvements.append(meta.get('eq_snr_improvement_db', 0))
        if 'alpha_ratio' in meta or 'eq_alpha_ratio' in meta:
            alpha_ratios.append(meta.get('alpha_ratio', meta.get('eq_alpha_ratio', 0)))
        if 'eq_pattern_preservation' in meta:
            preservations.append(meta.get('eq_pattern_preservation', 0))
    
    print("\n" + "="*70)
    print("📊 Statistics:")
    if snr_improvements:
        print(f"  SNR Improvement: mean={np.mean(snr_improvements):.2f} dB, "
              f"median={np.median(snr_improvements):.2f} dB, "
              f"std={np.std(snr_improvements):.2f} dB")
        print(f"    Range: [{np.min(snr_improvements):.2f}, {np.max(snr_improvements):.2f}] dB")
        print(f"    Samples with ΔSNR ≥ 4 dB: {np.sum(np.array(snr_improvements) >= 4.0)}/{len(snr_improvements)}")
    
    if alpha_ratios:
        print(f"  Alpha Ratio: mean={np.mean(alpha_ratios):.3f}x, "
              f"median={np.median(alpha_ratios):.3f}x")
        print(f"    Range: [{np.min(alpha_ratios):.3f}x, {np.max(alpha_ratios):.3f}x]")
        print(f"    Samples in [0.1x, 3x]: {np.sum((np.array(alpha_ratios) >= 0.1) & (np.array(alpha_ratios) <= 3.0))}/{len(alpha_ratios)}")
    
    if preservations:
        print(f"  Pattern Preservation: mean={np.mean(preservations):.3f}, "
              f"median={np.median(preservations):.3f}")
        print(f"    Samples with pres ≥ 0.5: {np.sum(np.array(preservations) >= 0.5)}/{len(preservations)}")
    
    print("="*70)

if __name__ == '__main__':
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/dataset_scenario_b_3840.pkl'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    print_metadata(dataset_path, num_samples)

