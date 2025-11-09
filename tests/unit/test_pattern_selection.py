"""
Unit tests for pattern selection and injection logic.
"""
import pytest
import numpy as np
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.covert_injection_phase1 import inject_covert_channel_fixed_phase1


@pytest.mark.unit
def test_pattern_selection_mid(mock_resource_grid, sample_ofdm_grid):
    """Test pattern selection with 'mid' subband mode."""
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='mid',
        covert_amp=0.5
    )
    
    assert 'selected_subcarriers' in injection_info
    assert 'selected_symbols' in injection_info
    assert injection_info['subband_mode'] == 'mid'
    assert len(injection_info['selected_subcarriers']) == 16
    assert all(24 <= sc < 40 for sc in injection_info['selected_subcarriers'])


@pytest.mark.unit
def test_pattern_selection_random16(mock_resource_grid, sample_ofdm_grid):
    """Test pattern selection with 'random16' subband mode."""
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='random16',
        covert_amp=0.5
    )
    
    assert 'selected_subcarriers' in injection_info
    assert injection_info['subband_mode'] == 'random16'
    assert len(injection_info['selected_subcarriers']) == 16
    # Check that subcarriers are contiguous
    selected = sorted(injection_info['selected_subcarriers'])
    assert selected[-1] - selected[0] == 15, "Subcarriers should be contiguous"


@pytest.mark.unit
def test_pattern_selection_hopping(mock_resource_grid, sample_ofdm_grid):
    """Test pattern selection with 'hopping' subband mode."""
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='hopping',
        covert_amp=0.5
    )
    
    assert 'selected_subcarriers' in injection_info
    assert injection_info['subband_mode'] == 'hopping'
    assert 'selected_subcarriers_per_symbol' in injection_info
    assert injection_info['selected_subcarriers_per_symbol'] is not None


@pytest.mark.unit
def test_pattern_selection_sparse(mock_resource_grid, sample_ofdm_grid):
    """Test pattern selection with 'sparse' subband mode."""
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='sparse',
        covert_amp=0.5
    )
    
    assert 'selected_subcarriers' in injection_info
    assert injection_info['subband_mode'] == 'sparse'
    assert len(injection_info['selected_subcarriers']) == 16
    # Check that subcarriers are NOT all contiguous
    selected = sorted(injection_info['selected_subcarriers'])
    gaps = [selected[i+1] - selected[i] for i in range(len(selected)-1)]
    assert any(gap > 1 for gap in gaps), "Sparse pattern should have gaps"


@pytest.mark.unit
def test_injection_info_structure(mock_resource_grid, sample_ofdm_grid):
    """Test that injection_info has all required fields."""
    tx_grid, injection_info = inject_covert_channel_fixed_phase1(
        sample_ofdm_grid.copy(),
        mock_resource_grid,
        pattern='fixed',
        subband_mode='mid',
        covert_amp=0.5
    )
    
    required_fields = [
        'pattern',
        'subband_mode',
        'selected_subcarriers',
        'selected_symbols',
        'num_covert_subs',
        'num_covert_syms',
        'covert_amp',
    ]
    
    for field in required_fields:
        assert field in injection_info, f"Missing required field: {field}"

