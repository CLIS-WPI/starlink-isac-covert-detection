"""
Pytest configuration and shared fixtures for all tests.
"""
import os
import sys
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def workspace_root():
    """Return the workspace root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(workspace_root):
    """Create a temporary directory for test data."""
    test_dir = workspace_root / "test_pipeline_data"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after all tests
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="function")
def clean_test_env(workspace_root):
    """Clean test environment before each test."""
    test_dir = workspace_root / "test_pipeline_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after test
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def mock_resource_grid():
    """Create a mock resource grid for testing."""
    class MockResourceGrid:
        def __init__(self):
            self.num_effective_subcarriers = 64
            self.num_symbols = 10
    
    return MockResourceGrid()


@pytest.fixture
def sample_ofdm_grid():
    """Create a sample OFDM grid for testing."""
    return np.random.randn(1, 1, 1, 10, 64).astype(np.complex64) * (1 + 1j)


@pytest.fixture
def sample_injection_info():
    """Create sample injection info for testing."""
    return {
        'pattern': 'fixed',
        'subband_mode': 'mid',
        'selected_subcarriers': list(range(24, 40)),
        'selected_symbols': [0, 2, 4, 6, 8],
        'num_covert_subs': 16,
        'num_covert_syms': 5,
        'covert_amp': 0.5,
    }


@pytest.fixture
def pattern_configs():
    """Return all pattern configurations to test."""
    return [
        {'pattern': 'fixed', 'subband': 'mid', 'name': 'contiguous'},
        {'pattern': 'fixed', 'subband': 'random16', 'name': 'random'},
        {'pattern': 'fixed', 'subband': 'hopping', 'name': 'hopping'},
        {'pattern': 'fixed', 'subband': 'sparse', 'name': 'sparse'},
    ]


@pytest.fixture
def scenario_configs():
    """Return scenario configurations for testing."""
    return {
        'scenario_a': {
            'scenario': 'sat',
            'snr_list': '15,20',
            'covert_amp_list': '0.5',
        },
        'scenario_b': {
            'scenario': 'ground',
            'snr_list': '15,20',
            'covert_amp_list': '0.5',
        },
    }

