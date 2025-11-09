#!/usr/bin/env python3
"""
🔍 Test SNR Measurement and EQ with H_true
===========================================
Tests SNR measurement correctness and upper bound with H_true.
"""

import pickle
import numpy as np
from core.csi_estimation import mmse_equalize, estimate_csi_lmmse_2d_separable

def test_snr_with_h_true(dataset_path, num_samples=10):
    """Test SNR improvement with H_true (upper bound test)."""
    
    print("="*70)
    print("🔍 Test SNR Measurement with H_true (Upper Bound)")
    print("="*70)
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    tx_grids = dataset.get('tx_grids', [])
    rx_grids = dataset.get('rx_grids', [])
    meta_list = dataset.get('meta', [])
    
    if len(tx_grids) == 0 or len(rx_grids) == 0:
        print("❌ No data found in dataset!")
        return
    
    print(f"\n📊 Testing {min(num_samples, len(rx_grids))} samples...\n")
    
    results = []
    
    for i in range(min(num_samples, len(rx_grids))):
        tx_grid = tx_grids[i]
        rx_grid = rx_grids[i]
        meta = meta_list[i] if i < len(meta_list) else {}
        
        if isinstance(meta, tuple):
            _, meta = meta
        
        # Fix tx_grid shape if needed (should be [symbols, subcarriers])
        if tx_grid.ndim == 5:
            # [1, 1, 1, symbols, subcarriers] -> [symbols, subcarriers]
            tx_grid = np.squeeze(tx_grid, axis=(0, 1, 2))
        elif tx_grid.ndim == 3:
            # [1, symbols, subcarriers] -> [symbols, subcarriers]
            tx_grid = np.squeeze(tx_grid, axis=0)
        elif tx_grid.ndim == 4:
            # [1, 1, symbols, subcarriers] -> [symbols, subcarriers]
            tx_grid = np.squeeze(tx_grid, axis=(0, 1))
        
        # Ensure rx_grid is 2D
        if rx_grid.ndim != 2:
            rx_grid = np.squeeze(rx_grid)
        
        # Get true channel (if available)
        h_true = meta.get('h_true_dl')
        if h_true is None:
            print(f"  Sample {i}: ⚠️  No H_true available, skipping...")
            continue
        
        # Convert to numpy if needed
        if hasattr(h_true, 'numpy'):
            h_true = h_true.numpy()
        h_true = np.array(h_true, dtype=np.complex64)
        
        # Ensure h_true has correct shape (should match rx_grid: [symbols, subcarriers])
        if h_true.ndim == 1:
            # If 1D, assume it's [subcarriers] and needs to be tiled to [symbols, subcarriers]
            num_symbols = rx_grid.shape[0]
            h_true = np.tile(h_true[np.newaxis, :], (num_symbols, 1))
        elif h_true.ndim == 2:
            # Handle different 2D shapes
            if h_true.shape == (rx_grid.shape[1], 1) or h_true.shape == (1, rx_grid.shape[1]):
                # h_true is [subcarriers, 1] or [1, subcarriers] - transpose and tile
                if h_true.shape[0] == rx_grid.shape[1]:
                    h_true = h_true.T  # [1, subcarriers]
                h_true = np.tile(h_true, (rx_grid.shape[0], 1))  # [symbols, subcarriers]
            elif h_true.shape[0] != rx_grid.shape[0]:
                # Shape mismatch - try to fix
                if h_true.shape[1] == rx_grid.shape[1]:
                    # Same subcarriers, tile symbols
                    num_symbols = rx_grid.shape[0]
                    h_true = np.tile(h_true[0:1, :], (num_symbols, 1))
                elif h_true.shape[0] == rx_grid.shape[1]:
                    # h_true is [subcarriers, ?] - transpose
                    h_true = h_true.T
                    if h_true.shape[0] != rx_grid.shape[0]:
                        h_true = np.tile(h_true[0:1, :], (rx_grid.shape[0], 1))
                else:
                    print(f"  Sample {i}: ⚠️  H_true shape mismatch: {h_true.shape} vs {rx_grid.shape}, skipping...")
                    continue
        
        # Get SNR input
        snr_input = meta.get('snr_dl', 20.0)
        
        # Test with H_true
        rx_eq_true, eq_info_true = mmse_equalize(
            rx_grid, h_true,
            snr_db=snr_input,
            alpha_reg=None,
            blend_factor=1.0,  # Full equalization
            noise_variance_est=None
        )
        
        # Test with H_est (LMMSE)
        h_est, csi_info = estimate_csi_lmmse_2d_separable(
            tx_grid, rx_grid,
            pilot_symbols=[2, 7],
            metadata=meta,
            csi_cfg=None
        )
        
        rx_eq_est, eq_info_est = mmse_equalize(
            rx_grid, h_est,
            snr_db=snr_input,
            alpha_reg=None,
            blend_factor=1.0,  # Full equalization
            noise_variance_est=csi_info.get('noise_variance')
        )
        
        # Compare results
        snr_imp_true = eq_info_true.get('snr_improvement_db', 0)
        snr_imp_est = eq_info_est.get('snr_improvement_db', 0)
        
        results.append({
            'sample': i,
            'snr_input': snr_input,
            'snr_raw': eq_info_true.get('snr_raw_db', 0),
            'snr_eq_true': eq_info_true.get('snr_eq_db', 0),
            'snr_imp_true': snr_imp_true,
            'snr_eq_est': eq_info_est.get('snr_eq_db', 0),
            'snr_imp_est': snr_imp_est,
            'alpha_true': eq_info_true.get('alpha_used', 0),
            'alpha_est': eq_info_est.get('alpha_used', 0),
            'h_power_true': np.mean(np.abs(h_true)**2),
            'h_power_est': np.mean(np.abs(h_est)**2),
        })
        
        print(f"  Sample {i}: SNR_input={snr_input:.1f} dB")
        print(f"    SNR_raw: {eq_info_true.get('snr_raw_db', 0):.2f} dB")
        print(f"    With H_true: SNR_eq={eq_info_true.get('snr_eq_db', 0):.2f} dB, ΔSNR={snr_imp_true:.2f} dB")
        print(f"    With H_est:  SNR_eq={eq_info_est.get('snr_eq_db', 0):.2f} dB, ΔSNR={snr_imp_est:.2f} dB")
        print()
    
    # Summary
    if results:
        snr_imp_true_mean = np.mean([r['snr_imp_true'] for r in results])
        snr_imp_est_mean = np.mean([r['snr_imp_est'] for r in results])
        
        print("="*70)
        print("📊 Summary:")
        print(f"  Mean SNR improvement with H_true: {snr_imp_true_mean:.2f} dB")
        print(f"  Mean SNR improvement with H_est:  {snr_imp_est_mean:.2f} dB")
        print()
        
        if snr_imp_true_mean >= 5.0:
            print("  ✅ Upper bound test PASSED: ΔSNR ≥ 5 dB with H_true")
            print("     → Problem is in H estimation, not SNR measurement")
        else:
            print("  ❌ Upper bound test FAILED: ΔSNR < 5 dB even with H_true")
            print("     → Problem is in SNR measurement or EQ definition")
        
        if snr_imp_est_mean >= 3.0:
            print("  ✅ H_est test PASSED: ΔSNR ≥ 3 dB with estimated H")
        else:
            print("  ❌ H_est test FAILED: ΔSNR < 3 dB with estimated H")
        print("="*70)

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/dataset_scenario_b_3840.pkl'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    test_snr_with_h_true(dataset_path, num_samples)

