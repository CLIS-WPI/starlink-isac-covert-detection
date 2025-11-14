"""
End-to-End Test for Scenario A (Single-hop Downlink)
"""
import pytest
import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Note: We use subprocess to call generate_dataset_parallel.py
# because main() doesn't accept arguments directly


@pytest.mark.e2e
@pytest.mark.slow
def test_scenario_a_generation(clean_test_env):
    """Test Scenario A: Single-hop dataset generation."""
    # Generate dataset using subprocess
    import subprocess
    try:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cmd = [
            'python3', 'generate_dataset_parallel.py',
            '--scenario', 'sat',
            '--total-samples', '50',
            '--snr-list', '15,20',
            '--covert-amp-list', '0.5',
            '--samples-per-config', '12',
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workspace_root
        )
        
        if result.returncode != 0:
            pytest.fail(f"Dataset generation failed: {result.stderr[:500]}")
        
        # Check if dataset was generated (in default dataset directory)
        dataset_dir = Path(workspace_root) / 'dataset'
        dataset_files = list(dataset_dir.glob('dataset_scenario_a_*.pkl'))
        
        # Also check test env directory
        if len(dataset_files) == 0:
            test_dataset_dir = Path(clean_test_env / 'dataset')
            dataset_files = list(test_dataset_dir.glob('dataset_scenario_a_*.pkl'))
        
        assert len(dataset_files) > 0, "No Scenario A dataset files found"
        
        # Load and verify dataset
        import pickle
        with open(dataset_files[0], 'rb') as f:
            dataset = pickle.load(f)
        
        # Dataset structure: rx_grids, labels, meta (not 'data')
        assert 'rx_grids' in dataset or 'data' in dataset, "Dataset missing 'rx_grids' or 'data' key"
        assert 'labels' in dataset, "Dataset missing 'labels' key"
        assert 'meta' in dataset, "Dataset missing 'meta' key"
        
        # Check data exists
        if 'rx_grids' in dataset:
            assert len(dataset['rx_grids']) > 0, "Dataset is empty"
        elif 'data' in dataset:
            assert len(dataset['data']) > 0, "Dataset is empty"
        
        print(f"✅ Scenario A dataset generated: {dataset_files[0].name}")
        num_samples = len(dataset.get('rx_grids', dataset.get('data', [])))
        print(f"   Samples: {num_samples}")
        
    except Exception as e:
        pytest.fail(f"Scenario A generation failed: {e}")


@pytest.mark.e2e
@pytest.mark.slow
def test_scenario_a_metadata(clean_test_env):
    """Test Scenario A metadata structure."""
    # This test would verify metadata structure
    # For now, we'll skip if no dataset exists
    dataset_dir = clean_test_env / 'dataset'
    dataset_files = list(dataset_dir.glob('dataset_scenario_a_*.pkl'))
    
    if len(dataset_files) == 0:
        pytest.skip("No Scenario A dataset found. Run test_scenario_a_generation first.")
    
    import pickle
    with open(dataset_files[0], 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    assert len(meta_list) > 0, "No metadata found"
    
    # Check metadata structure
    for meta in meta_list:
        if isinstance(meta, tuple):
            _, meta = meta
        
        # Scenario A should have these keys
        assert 'scenario' in meta or 'scenario_type' in meta, "Missing scenario info"
        assert 'snr_db' in meta, "Missing SNR info"

