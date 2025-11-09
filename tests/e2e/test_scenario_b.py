"""
End-to-End Test for Scenario B (Dual-hop Relay)
"""
import pytest
import sys
import os
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("pattern_config", [
    {'pattern': 'fixed', 'subband': 'mid', 'name': 'contiguous'},
    {'pattern': 'fixed', 'subband': 'random16', 'name': 'random'},
    {'pattern': 'fixed', 'subband': 'hopping', 'name': 'hopping'},
    {'pattern': 'fixed', 'subband': 'sparse', 'name': 'sparse'},
])
def test_scenario_b_pattern_generation(clean_test_env, pattern_config):
    """Test Scenario B dataset generation with different patterns."""
    import subprocess
    import pickle
    
    # Generate dataset with specific pattern
    cmd = [
        'python3', 'generate_dataset_parallel.py',
        '--scenario', 'ground',
        '--total-samples', '30',
        '--snr-list', '15,20',
        '--covert-amp-list', '0.5',
        '--samples-per-config', '7',
        '--pattern', pattern_config['pattern'],
        '--subband', pattern_config['subband'],
    ]
    
    try:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workspace_root
        )
        
        if result.returncode != 0:
            pytest.fail(f"Dataset generation failed: {result.stderr[:200]}")
        
        # Find generated dataset
        dataset_dir = Path('dataset')
        dataset_files = list(dataset_dir.glob('dataset_scenario_b_*.pkl'))
        
        if len(dataset_files) == 0:
            pytest.fail("No Scenario B dataset files found")
        
        latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
        
        # Load and verify dataset
        with open(latest, 'rb') as f:
            dataset = pickle.load(f)
        
        assert 'data' in dataset, "Dataset missing 'data' key"
        assert 'meta' in dataset, "Dataset missing 'meta' key"
        
        # Verify injection_info
        meta_list = dataset.get('meta', [])
        injection_info_count = 0
        
        for meta in meta_list:
            if isinstance(meta, tuple):
                _, meta = meta
            
            if 'injection_info' in meta:
                injection_info = meta['injection_info']
                if injection_info.get('subband_mode') == pattern_config['subband']:
                    injection_info_count += 1
        
        assert injection_info_count > 0, f"No injection_info found for {pattern_config['name']}"
        
        print(f"✅ {pattern_config['name']}: {len(meta_list)} samples, {injection_info_count} with injection_info")
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timeout generating dataset for {pattern_config['name']}")
    except Exception as e:
        pytest.fail(f"Error generating dataset for {pattern_config['name']}: {e}")


@pytest.mark.e2e
@pytest.mark.slow
def test_scenario_b_eq_performance(clean_test_env):
    """Test Scenario B EQ performance metrics."""
    import pickle
    
    dataset_dir = Path('dataset')
    dataset_files = list(dataset_dir.glob('dataset_scenario_b_*.pkl'))
    
    if len(dataset_files) == 0:
        pytest.skip("No Scenario B dataset found. Run test_scenario_b_pattern_generation first.")
    
    latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest, 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    
    preservations = []
    snr_improvements = []
    
    for meta in meta_list:
        if isinstance(meta, tuple):
            _, meta = meta
        
        if 'eq_pattern_preservation' in meta:
            preservations.append(meta['eq_pattern_preservation'])
        if 'eq_snr_improvement_db' in meta:
            snr_improvements.append(meta['eq_snr_improvement_db'])
    
    if len(preservations) > 0:
        median_preservation = np.median(preservations)
        assert median_preservation >= 0.45, f"Pattern preservation too low: {median_preservation}"
        print(f"✅ Pattern preservation: median={median_preservation:.3f}")
    
    if len(snr_improvements) > 0:
        mean_snr_improvement = np.mean(snr_improvements)
        assert mean_snr_improvement >= 25.0, f"SNR improvement too low: {mean_snr_improvement}"
        print(f"✅ SNR improvement: mean={mean_snr_improvement:.2f} dB")

