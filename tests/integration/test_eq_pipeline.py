"""
Integration tests for EQ pipeline (CSI estimation + MMSE equalization).
"""
import pytest
import numpy as np
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.csi_estimation import mmse_equalize, compute_pattern_preservation
from core.covert_injection_phase1 import inject_covert_channel_fixed_phase1


@pytest.mark.integration
def test_eq_pipeline_with_injection(mock_resource_grid, sample_ofdm_grid):
    """Test complete EQ pipeline with injection."""
    # Inject covert pattern
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='mid',
        covert_amp=0.5
    )
    
    # Simulate channel and noise
    num_symbols, num_subcarriers = tx_grid.shape[-2], tx_grid.shape[-1]
    h_true = np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * (0.5 + 0.5j)
    
    # Create received signal (with channel and noise)
    rx_grid = tx_grid * h_true + np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * 0.1
    
    # Equalize
    x_eq, eq_info = mmse_equalize(
        rx_grid,
        h_true,  # Use true channel for testing
        snr_db=20.0,
        metadata={'injection_info': injection_info}
    )
    
    # Check EQ info
    assert 'snr_improvement_db' in eq_info
    assert 'alpha_used' in eq_info
    
    # Compute pattern preservation
    target_subcarriers = np.array(injection_info['selected_subcarriers'])
    
    # Ensure proper shape for compute_pattern_preservation
    # It expects (symbols, subcarriers) shape
    if tx_grid.ndim == 3:
        tx_grid = tx_grid[0]  # Take first sample if batch dimension exists
    if rx_grid.ndim == 3:
        rx_grid = rx_grid[0]
    if x_eq.ndim == 3:
        x_eq = x_eq[0]
    
    # Ensure 2D shape
    if tx_grid.ndim == 1:
        tx_grid = tx_grid.reshape(1, -1)
    if rx_grid.ndim == 1:
        rx_grid = rx_grid.reshape(1, -1)
    if x_eq.ndim == 1:
        x_eq = x_eq.reshape(1, -1)
    
    metrics = compute_pattern_preservation(
        tx_grid,
        rx_grid,
        x_eq,
        target_subcarriers=target_subcarriers
    )
    
    assert 'correlation_eq' in metrics
    assert 'energy_ratio_eq' in metrics


@pytest.mark.integration
def test_adaptive_band_emphasis(mock_resource_grid, sample_ofdm_grid):
    """Test that adaptive band emphasis works with different patterns."""
    patterns = ['mid', 'random16', 'sparse']
    
    for subband_mode in patterns:
        tx_grid, injection_info = inject_covert_channel_fixed_phase1(
            sample_ofdm_grid.copy(),
            mock_resource_grid,
            pattern='fixed',
            subband_mode=subband_mode,
            covert_amp=0.5
        )
        
        num_symbols, num_subcarriers = tx_grid.shape[-2], tx_grid.shape[-1]
        h_est = np.ones((num_symbols, num_subcarriers), dtype=np.complex64) * (0.5 + 0.5j)
        rx_grid = tx_grid * h_est
        
        # Equalize with injection_info
        x_eq, eq_info = mmse_equalize(
            rx_grid,
            h_est,
            snr_db=20.0,
            metadata={'injection_info': injection_info}
        )
        
        # Should not fail
        assert x_eq.shape == rx_grid.shape

