"""
Integration tests for dataset generation pipeline.
"""
import pytest
import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.integration
@pytest.mark.slow
def test_dataset_structure(clean_test_env):
    """Test that generated dataset has correct structure."""
    # This test would require a dataset to be generated first
    # For now, we'll check if any dataset exists
    dataset_dir = Path('dataset')
    dataset_files = list(dataset_dir.glob('dataset_scenario_*.pkl'))
    
    if len(dataset_files) == 0:
        pytest.skip("No dataset files found. Generate dataset first.")
    
    # Load first dataset
    with open(dataset_files[0], 'rb') as f:
        dataset = pickle.load(f)
    
    # Check structure
    assert 'data' in dataset, "Dataset missing 'data' key"
    assert 'meta' in dataset, "Dataset missing 'meta' key"
    
    data = dataset['data']
    meta = dataset['meta']
    
    assert len(data) > 0, "Dataset data is empty"
    assert len(meta) > 0, "Dataset metadata is empty"
    assert len(data) == len(meta), "Data and metadata should have same length"
    
    # Check data shape
    sample = data[0]
    assert isinstance(sample, np.ndarray), "Data should be numpy arrays"
    assert len(sample.shape) >= 2, "Data should be at least 2D"


@pytest.mark.integration
def test_metadata_injection_info(clean_test_env):
    """Test that metadata contains injection_info."""
    dataset_dir = Path('dataset')
    dataset_files = list(dataset_dir.glob('dataset_scenario_b_*.pkl'))
    
    if len(dataset_files) == 0:
        pytest.skip("No Scenario B dataset found.")
    
    with open(dataset_files[0], 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    injection_info_count = 0
    
    for meta in meta_list:
        if isinstance(meta, tuple):
            _, meta = meta
        
        if 'injection_info' in meta:
            injection_info = meta['injection_info']
            assert 'selected_subcarriers' in injection_info
            assert 'selected_symbols' in injection_info
            assert 'subband_mode' in injection_info
            injection_info_count += 1
    
    assert injection_info_count > 0, "No injection_info found in metadata"


@pytest.mark.integration
def test_eq_metadata(clean_test_env):
    """Test that EQ metadata is present in Scenario B datasets."""
    dataset_dir = Path('dataset')
    dataset_files = list(dataset_dir.glob('dataset_scenario_b_*.pkl'))
    
    if len(dataset_files) == 0:
        pytest.skip("No Scenario B dataset found.")
    
    with open(dataset_files[0], 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    eq_metadata_count = 0
    
    for meta in meta_list:
        if isinstance(meta, tuple):
            _, meta = meta
        
        # Check for EQ-related metadata
        if 'eq_snr_improvement_db' in meta or 'eq_pattern_preservation' in meta:
            eq_metadata_count += 1
    
    # At least some samples should have EQ metadata
    assert eq_metadata_count > 0, "No EQ metadata found"

