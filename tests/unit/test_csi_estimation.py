"""
Unit tests for CSI estimation and MMSE equalization.
"""
import pytest
import numpy as np
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.csi_estimation import mmse_equalize, compute_pattern_preservation


@pytest.mark.unit
def test_mmse_equalize_basic():
    """Test basic MMSE equalization."""
    # Create simple test data
    num_symbols = 10
    num_subcarriers = 64
    
    # Create channel estimate (simple)
    h_est = np.ones((num_symbols, num_subcarriers), dtype=np.complex64) * (0.5 + 0.5j)
    
    # Create received signal
    rx_grid = np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * (1 + 1j)
    
    # Equalize
    x_eq, eq_info = mmse_equalize(
        rx_grid,
        h_est,
        snr_db=20.0,
        alpha_reg=None,
        blend_factor=1.0
    )
    
    assert x_eq.shape == rx_grid.shape, "Equalized signal should have same shape"
    assert 'snr_raw_db' in eq_info, "EQ info should contain snr_raw_db"
    assert 'snr_eq_db' in eq_info, "EQ info should contain snr_eq_db"
    assert 'snr_improvement_db' in eq_info, "EQ info should contain snr_improvement_db"


@pytest.mark.unit
def test_mmse_equalize_with_metadata(sample_injection_info):
    """Test MMSE equalization with injection_info metadata."""
    num_symbols = 10
    num_subcarriers = 64
    
    h_est = np.ones((num_symbols, num_subcarriers), dtype=np.complex64) * (0.5 + 0.5j)
    rx_grid = np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * (1 + 1j)
    
    metadata = {
        'injection_info': sample_injection_info
    }
    
    x_eq, eq_info = mmse_equalize(
        rx_grid,
        h_est,
        snr_db=20.0,
        metadata=metadata
    )
    
    assert x_eq.shape == rx_grid.shape
    assert 'alpha_used' in eq_info


@pytest.mark.unit
def test_compute_pattern_preservation():
    """Test pattern preservation computation."""
    num_symbols = 10
    num_subcarriers = 64
    
    # Create test grids
    tx_grid = np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * (1 + 1j)
    rx_raw = tx_grid * 0.5  # Simulate attenuation
    rx_eq = tx_grid * 0.9   # Simulate better recovery after EQ
    
    target_subcarriers = np.arange(24, 40)
    
    metrics = compute_pattern_preservation(
        tx_grid,
        rx_raw,
        rx_eq,
        target_subcarriers=target_subcarriers
    )
    
    assert 'correlation_eq' in metrics
    assert 'energy_ratio_eq' in metrics
    assert 'correlation_improvement' in metrics
    
    # After EQ, correlation should be better than raw
    assert metrics['correlation_eq'] >= metrics['correlation_raw'], \
        "EQ should improve correlation"


@pytest.mark.unit
def test_alpha_ratio_calculation():
    """Test that alpha_ratio is calculated correctly."""
    num_symbols = 10
    num_subcarriers = 64
    
    h_est = np.ones((num_symbols, num_subcarriers), dtype=np.complex64) * (0.5 + 0.5j)
    rx_grid = np.random.randn(num_symbols, num_subcarriers).astype(np.complex64) * (1 + 1j)
    
    x_eq, eq_info = mmse_equalize(
        rx_grid,
        h_est,
        snr_db=20.0
    )
    
    if 'alpha_ratio' in eq_info and 'h_power_mean' in eq_info:
        h_power = eq_info['h_power_mean']
        alpha = eq_info['alpha_used']
        
        if h_power > 0:
            expected_ratio = alpha / h_power
            assert abs(eq_info['alpha_ratio'] - expected_ratio) < 1e-6, \
                "alpha_ratio should be alpha / h_power"

