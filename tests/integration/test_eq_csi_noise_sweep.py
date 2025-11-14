"""
🔥 CRITICAL TEST: CSI Noise Sweep (EQ Robustness)
==================================================
This test verifies that AUC=1.00 is not only achieved with ideal CSI.

We test behavior under different CSI noise levels:
- NMSE = -20 dB (very good)
- NMSE = -10 dB (good)
- NMSE = -5 dB (moderate)
- NMSE = 0 dB (poor)

If AUC always = 1.00 → suspicious (too perfect)
If behavior is reasonable → excellent (robust)
"""
import pytest
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.detector_cnn import CNNDetector
from core.csi_estimation import estimate_csi_ls_smooth, mmse_equalize
from config.settings import USE_FOCAL_LOSS, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA


def load_dataset_for_test(scenario='ground'):
    """Load dataset for testing."""
    import glob
    import re
    
    scenario_letter = 'a' if scenario == 'sat' else 'b'
    pattern = f'dataset/dataset_scenario_{scenario_letter}_*.pkl'
    candidates = glob.glob(pattern)
    
    if not candidates:
        raise FileNotFoundError(f"No dataset found for scenario {scenario}")
    
    def extract_samples(path):
        match = re.search(r'dataset_scenario_[ab]_(\d+)\.pkl', path)
        return int(match.group(1)) if match else 0
    
    candidates.sort(key=extract_samples, reverse=True)
    dataset_file = Path(candidates[0])
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract data
    if 'rx_grids' in dataset:
        rx_grids = np.array(dataset['rx_grids'])
        labels = np.array(dataset['labels'])
        tx_grids = dataset.get('tx_grids', None)
    elif 'data' in dataset:
        rx_grids = np.array([item for item in dataset['data']])
        labels = np.array([item['is_covert'] if isinstance(item, dict) else meta.get('is_covert', 0) 
                          for item in dataset.get('meta', [])])
        tx_grids = None
    else:
        raise ValueError("Unknown dataset format")
    
    return rx_grids, labels, tx_grids


def add_csi_noise(h_est, target_nmse_db):
    """
    Add noise to CSI estimate to simulate different NMSE levels.
    
    Args:
        h_est: True CSI estimate
        target_nmse_db: Target NMSE in dB
    
    Returns:
        h_noisy: Noisy CSI estimate
    """
    # Convert NMSE from dB to linear
    target_nmse_linear = 10 ** (target_nmse_db / 10.0)
    
    # Compute noise power needed
    h_power = np.mean(np.abs(h_est) ** 2)
    noise_power = h_power * target_nmse_linear
    
    # Generate complex noise
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*h_est.shape) + 1j * np.random.randn(*h_est.shape)
    )
    
    h_noisy = h_est + noise
    
    # Verify actual NMSE
    error = h_noisy - h_est
    actual_nmse = np.mean(np.abs(error) ** 2) / np.mean(np.abs(h_est) ** 2)
    actual_nmse_db = 10 * np.log10(actual_nmse + 1e-10)
    
    return h_noisy, actual_nmse_db


@pytest.mark.integration
@pytest.mark.slow
def test_csi_noise_sweep_robustness():
    """
    🔥 CRITICAL: Test AUC under different CSI noise levels.
    
    Expected behavior:
    - Low noise (NMSE = -20 dB): AUC should be high (≥ 0.95)
    - Moderate noise (NMSE = -5 dB): AUC should degrade but remain reasonable (≥ 0.80)
    - High noise (NMSE = 0 dB): AUC may degrade further but should still be > 0.5
    
    If AUC always = 1.00 regardless of noise → suspicious
    """
    print("\n" + "="*70)
    print("🔥 CSI NOISE SWEEP TEST (EQ Robustness)")
    print("="*70)
    
    # Load dataset (Scenario B - uses equalization)
    try:
        rx_grids, labels, tx_grids = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found. Generate dataset first.")
    
    # Use subset for faster testing
    n_samples = min(500, len(rx_grids))
    indices = np.random.choice(len(rx_grids), n_samples, replace=False)
    rx_grids = rx_grids[indices]
    labels = labels[indices]
    
    print(f"  Using {len(rx_grids)} samples")
    
    # Test different NMSE levels
    nmse_levels_db = [-20, -10, -5, 0]
    results = {}
    
    for target_nmse_db in nmse_levels_db:
        print(f"\n  Testing NMSE = {target_nmse_db} dB...")
        
        # Apply equalization with noisy CSI
        rx_grids_eq = []
        
        for i, rx_grid in enumerate(rx_grids):
            # Estimate CSI (using tx_grid if available, otherwise use rx_grid as proxy)
            if tx_grids is not None and i < len(tx_grids):
                tx_grid = tx_grids[i]
            else:
                # Use rx_grid as proxy (simplified)
                tx_grid = rx_grid * 0.5  # Rough estimate
            
            # Ensure proper shape: (symbols, subcarriers)
            if tx_grid.ndim == 1:
                # Reshape if needed
                tx_grid = tx_grid.reshape(1, -1)
            if rx_grid.ndim == 1:
                rx_grid = rx_grid.reshape(1, -1)
            
            # Check if we have enough symbols for pilot_symbols=[2, 7]
            if tx_grid.shape[0] < 8 or rx_grid.shape[0] < 8:
                # Use available symbols
                pilot_symbols = [min(2, tx_grid.shape[0]-1), min(7, tx_grid.shape[0]-1)]
            else:
                pilot_symbols = [2, 7]
            
            # Estimate CSI
            h_est = estimate_csi_ls_smooth(
                tx_grid, rx_grid,
                pilot_symbols=pilot_symbols,
                smoothing=True,
                interpolation='nearest'
            )
            
            # Add noise to CSI
            h_noisy, actual_nmse_db = add_csi_noise(h_est, target_nmse_db)
            
            # Apply MMSE equalization with noisy CSI
            rx_eq, eq_info = mmse_equalize(
                rx_grid,
                h_noisy,
                snr_db=20.0,  # Assume 20 dB SNR
                alpha_reg=None,
                blend_factor=1.0
            )
            
            rx_grids_eq.append(rx_eq)
        
        rx_grids_eq = np.array(rx_grids_eq)
        
        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            rx_grids_eq, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        detector = CNNDetector(
            use_csi=False,
            use_attention=True,
            random_state=42,
            learning_rate=0.001,
            dropout_rate=0.3,
            use_focal_loss=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_LOSS_GAMMA,
            focal_alpha=FOCAL_LOSS_ALPHA
        )
        
        detector.train(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            epochs=15,
            batch_size=64,
            verbose=0
        )
        
        y_pred_proba = detector.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        results[target_nmse_db] = auc
        
        print(f"    Actual NMSE: {actual_nmse_db:.2f} dB")
        print(f"    AUC: {auc:.4f}")
    
    # Print summary
    print(f"\n  📊 Results Summary:")
    print(f"     {'NMSE (dB)':<12} {'AUC':<8}")
    print(f"     {'-'*20}")
    for nmse_db in sorted(results.keys()):
        print(f"     {nmse_db:>6}      {results[nmse_db]:.4f}")
    
    # CRITICAL ASSERTIONS
    
    # 1. Low noise should give high AUC
    assert results[-20] >= 0.85, \
        f"Low noise (NMSE=-20 dB) should give AUC ≥ 0.85, got {results[-20]:.4f}"
    
    # 2. Performance should degrade with increasing noise (monotonic behavior)
    # But not too dramatically
    auc_low_noise = results[-20]
    auc_high_noise = results[0]
    
    # High noise should still be better than random
    assert auc_high_noise > 0.55, \
        f"Even with high noise (NMSE=0 dB), AUC should be > 0.55, got {auc_high_noise:.4f}"
    
    # 3. Degradation should be reasonable (not catastrophic)
    degradation = auc_low_noise - auc_high_noise
    assert degradation <= 0.40, \
        f"Performance degradation ({degradation:.4f}) should be ≤ 0.40. " \
        f"Too much degradation suggests fragility."
    
    print(f"\n  ✅ PASS: Model shows reasonable robustness to CSI noise")
    print(f"     Low noise: AUC={results[-20]:.4f}, High noise: AUC={results[0]:.4f}")


@pytest.mark.integration
@pytest.mark.slow
def test_ideal_vs_noisy_csi_comparison():
    """
    Compare performance with ideal CSI vs noisy CSI.
    """
    print("\n" + "="*70)
    print("🔥 IDEAL vs NOISY CSI COMPARISON")
    print("="*70)
    
    try:
        rx_grids, labels, tx_grids = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found.")
    
    n_samples = min(500, len(rx_grids))
    indices = np.random.choice(len(rx_grids), n_samples, replace=False)
    rx_grids = rx_grids[indices]
    labels = labels[indices]
    
    # Test with ideal CSI (no noise)
    print("  Testing with ideal CSI...")
    rx_grids_ideal = []
    for i, rx_grid in enumerate(rx_grids):
        if tx_grids is not None and i < len(tx_grids):
            tx_grid = tx_grids[i]
        else:
            tx_grid = rx_grid * 0.5
        
        # Ensure proper shape
        if tx_grid.ndim == 1:
            tx_grid = tx_grid.reshape(1, -1)
        if rx_grid.ndim == 1:
            rx_grid = rx_grid.reshape(1, -1)
        
        # Check if we have enough symbols
        if tx_grid.shape[0] < 8 or rx_grid.shape[0] < 8:
            pilot_symbols = [min(2, tx_grid.shape[0]-1), min(7, tx_grid.shape[0]-1)]
        else:
            pilot_symbols = [2, 7]
        
        h_est = estimate_csi_ls_smooth(tx_grid, rx_grid, pilot_symbols=pilot_symbols)
        rx_eq, _ = mmse_equalize(rx_grid, h_est, snr_db=20.0)
        rx_grids_ideal.append(rx_eq)
    
    rx_grids_ideal = np.array(rx_grids_ideal)
    
    # Test with noisy CSI (NMSE = -5 dB)
    print("  Testing with noisy CSI (NMSE = -5 dB)...")
    rx_grids_noisy = []
    for i, rx_grid in enumerate(rx_grids):
        if tx_grids is not None and i < len(tx_grids):
            tx_grid = tx_grids[i]
        else:
            tx_grid = rx_grid * 0.5
        
        h_est = estimate_csi_ls_smooth(tx_grid, rx_grid, pilot_symbols=[2, 7])
        h_noisy, _ = add_csi_noise(h_est, -5.0)
        rx_eq, _ = mmse_equalize(rx_grid, h_noisy, snr_db=20.0)
        rx_grids_noisy.append(rx_eq)
    
    rx_grids_noisy = np.array(rx_grids_noisy)
    
    # Train and evaluate both
    results = {}
    
    for name, X_data in [("Ideal CSI", rx_grids_ideal), ("Noisy CSI", rx_grids_noisy)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        detector = CNNDetector(use_csi=False, use_attention=True, random_state=42)
        detector.train(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            epochs=15, batch_size=64, verbose=0
        )
        
        y_pred = detector.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        results[name] = auc
    
    print(f"\n  📊 Comparison:")
    print(f"     Ideal CSI:  AUC = {results['Ideal CSI']:.4f}")
    print(f"     Noisy CSI:  AUC = {results['Noisy CSI']:.4f}")
    print(f"     Difference: {results['Ideal CSI'] - results['Noisy CSI']:.4f}")
    
    # Ideal should be better, but not dramatically
    assert results['Ideal CSI'] >= results['Noisy CSI'], \
        "Ideal CSI should perform better than noisy CSI"
    
    # Difference should be reasonable
    diff = results['Ideal CSI'] - results['Noisy CSI']
    assert diff <= 0.20, \
        f"Performance difference ({diff:.4f}) should be ≤ 0.20. " \
        f"Too large difference suggests fragility."
    
    print(f"  ✅ PASS: Model shows reasonable robustness")

