#!/usr/bin/env python3
"""
Ceiling Test with H_true
========================
Tests EQ performance with true channel (H_true) to establish upper bound.
Target: ΔSNR ≥ 5 dB, AUC >> 0.8
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, '/workspace')

from core.csi_estimation import mmse_equalize, compute_pattern_preservation
from core.isac_system import ISACSystem

def test_ceiling_with_h_true(num_samples=100):
    """Test EQ ceiling with H_true."""
    print("="*80)
    print("🔬 Ceiling Test with H_true")
    print("="*80)
    print(f"\nTarget: ΔSNR ≥ 5 dB, AUC >> 0.8")
    print(f"Testing {num_samples} samples...\n")
    
    # Load Scenario B dataset
    dataset_file = Path('dataset/dataset_scenario_b_5000.pkl')
    if not dataset_file.exists():
        print(f"❌ Dataset not found: {dataset_file}")
        return None
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    tx_grids = dataset['tx_grids']
    rx_grids = dataset['rx_grids']
    labels = dataset['labels']
    meta = dataset['meta']
    
    # Select samples
    indices = np.random.choice(len(tx_grids), min(num_samples, len(tx_grids)), replace=False)
    
    results = {
        'snr_improvements': [],
        'preservations': [],
        'h_true_available': 0,
        'h_est_used': 0,
    }
    
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{len(indices)}...")
        
        tx_grid = tx_grids[idx]
        rx_grid = rx_grids[idx]
        label = labels[idx]
        
        # Reshape grids if needed
        if len(tx_grid.shape) > 2:
            # (1, 1, 1, 10, 64) -> (10, 64)
            tx_grid = np.squeeze(tx_grid)
        if len(rx_grid.shape) > 2:
            rx_grid = np.squeeze(rx_grid)
        
        # Get metadata
        meta_entry = meta[idx]
        if isinstance(meta_entry, tuple):
            _, meta_entry = meta_entry
        
        # Check for H_true
        h_true = meta_entry.get('h_true_dl', None)
        injection_info = meta_entry.get('injection_info', {})
        
        if h_true is not None:
            results['h_true_available'] += 1
            
            # Reshape h_true if needed
            h_true = np.array(h_true)
            if h_true.shape != rx_grid.shape:
                if len(h_true.shape) == 2 and h_true.shape[1] == 1:
                    # (64, 1) -> (10, 64)
                    h_true = np.tile(h_true.T, (rx_grid.shape[0], 1))
                elif len(h_true.shape) == 1:
                    # (64,) -> (10, 64)
                    h_true = np.tile(h_true[np.newaxis, :], (rx_grid.shape[0], 1))
                elif len(h_true.shape) == 2 and h_true.shape[0] == rx_grid.shape[0]:
                    # Already correct shape
                    pass
                else:
                    print(f"  ⚠️  Skipping sample {idx}: h_true shape mismatch {h_true.shape} vs {rx_grid.shape}")
                    continue
            
            # Equalize with H_true
            snr_db = meta_entry.get('snr_db', 20.0)
            
            rx_eq, eq_info = mmse_equalize(
                rx_grid,
                h_true,  # Use true channel
                snr_db=snr_db,
                metadata={'injection_info': injection_info}
            )
            
            snr_improvement = eq_info.get('snr_improvement_db', 0)
            results['snr_improvements'].append(snr_improvement)
            
            # Compute pattern preservation
            target_subcarriers = np.arange(24, 40)  # Default
            if 'selected_subcarriers' in injection_info:
                target_subcarriers = np.array(injection_info['selected_subcarriers'], dtype=int)
            
            preservation_metrics = compute_pattern_preservation(
                tx_grid,
                rx_grid,
                rx_eq,
                target_subcarriers=target_subcarriers
            )
            
            preservation = preservation_metrics.get('correlation_eq', 0)
            results['preservations'].append(preservation)
        else:
            # Use estimated channel (for comparison)
            results['h_est_used'] += 1
    
    # Report results
    print("\n" + "="*80)
    print("📊 Ceiling Test Results")
    print("="*80)
    
    if len(results['snr_improvements']) > 0:
        snr_improvements = np.array(results['snr_improvements'])
        preservations = np.array(results['preservations'])
        
        print(f"\n✅ Samples with H_true: {results['h_true_available']}")
        print(f"⚠️  Samples with H_est: {results['h_est_used']}")
        
        print(f"\n📈 SNR Improvement:")
        print(f"   Mean: {np.mean(snr_improvements):.2f} ± {np.std(snr_improvements):.2f} dB")
        print(f"   Median: {np.median(snr_improvements):.2f} dB")
        print(f"   Min: {np.min(snr_improvements):.2f} dB")
        print(f"   Max: {np.max(snr_improvements):.2f} dB")
        print(f"   Target: ≥ 5 dB")
        print(f"   Status: {'✅ PASS' if np.mean(snr_improvements) >= 5.0 else '❌ FAIL'}")
        
        print(f"\n📈 Pattern Preservation:")
        print(f"   Mean: {np.mean(preservations):.3f} ± {np.std(preservations):.3f}")
        print(f"   Median: {np.median(preservations):.3f}")
        print(f"   Target: ≥ 0.5")
        print(f"   Status: {'✅ PASS' if np.median(preservations) >= 0.5 else '⚠️  CLOSE'}")
        
        print(f"\n💡 Conclusion:")
        if np.mean(snr_improvements) >= 5.0:
            print(f"   ✅ Ceiling test PASSED")
            print(f"   → EQ pipeline is correct")
            print(f"   → Problem is from Ĥ estimation quality")
        else:
            print(f"   ❌ Ceiling test FAILED")
            print(f"   → Problem is from EQ pipeline itself")
            print(f"   → Need to fix SNR measurement or EQ formula")
    else:
        print(f"\n❌ No samples with H_true found!")
        print(f"   → Need to regenerate dataset with H_true stored")
    
    return results

if __name__ == "__main__":
    results = test_ceiling_with_h_true(100)

