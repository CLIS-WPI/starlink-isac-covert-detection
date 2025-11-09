"""
End-to-End Test for Scenario A (Single-hop Downlink)
"""
import pytest
import sys
import os
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generate_dataset_parallel import main as generate_dataset_main


@pytest.mark.e2e
@pytest.mark.slow
def test_scenario_a_generation(clean_test_env):
    """Test Scenario A: Single-hop dataset generation."""
    import argparse
    
    # Create mock args for generate_dataset_parallel
    args = [
        '--scenario', 'sat',
        '--total-samples', '50',
        '--snr-list', '15,20',
        '--covert-amp-list', '0.5',
        '--samples-per-config', '12',
        '--output-dir', str(clean_test_env / 'dataset'),
    ]
    
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--total-samples', type=int, required=True)
    parser.add_argument('--snr-list', type=str, required=True)
    parser.add_argument('--covert-amp-list', type=str, required=True)
    parser.add_argument('--samples-per-config', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default='dataset')
    
    parsed_args = parser.parse_args(args)
    
    # Generate dataset
    try:
        generate_dataset_main([arg for arg in args if arg.startswith('--')])
        
        # Check if dataset was generated
        dataset_dir = Path(parsed_args.output_dir)
        dataset_files = list(dataset_dir.glob('dataset_scenario_a_*.pkl'))
        
        assert len(dataset_files) > 0, "No Scenario A dataset files found"
        
        # Load and verify dataset
        import pickle
        with open(dataset_files[0], 'rb') as f:
            dataset = pickle.load(f)
        
        assert 'data' in dataset, "Dataset missing 'data' key"
        assert 'meta' in dataset, "Dataset missing 'meta' key"
        assert len(dataset['data']) > 0, "Dataset is empty"
        
        print(f"✅ Scenario A dataset generated: {dataset_files[0].name}")
        print(f"   Samples: {len(dataset['data'])}")
        
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

