"""
Test cases for dataset generation
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.isac_system import ISACSystem
from core.dataset_generator import generate_dataset_multi_satellite


class TestDatasetGenerator:
    
    @pytest.fixture
    def isac_system(self):
        """Create ISAC system for testing."""
        return ISACSystem()
    
    def test_dataset_shape(self, isac_system):
        """Test that dataset has correct shapes."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=10,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )
        
        assert len(dataset['labels']) == 20  # 10 benign + 10 attack
        assert dataset['iq_samples'].shape[0] == 20
        assert dataset['csi'].shape[0] == 20
        assert dataset['radar_echo'].shape[0] == 20
        
    def test_power_ratio(self, isac_system):
        """Test that attack samples have higher power than benign."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=50,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )
        
        labels = dataset['labels']
        benign_idx = np.where(labels == 0)[0]
        attack_idx = np.where(labels == 1)[0]
        
        benign_power = np.mean([
            np.mean(np.abs(dataset['iq_samples'][i])**2) 
            for i in benign_idx
        ])
        attack_power = np.mean([
            np.mean(np.abs(dataset['iq_samples'][i])**2) 
            for i in attack_idx
        ])
        
        power_ratio = attack_power / benign_power
        
        # Attack should have at least 1.5× power
        assert power_ratio > 1.5, f"Power ratio too low: {power_ratio:.4f}"
        
    def test_emitter_locations(self, isac_system):
        """Test that attack samples have emitter locations."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=20,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )
        
        labels = dataset['labels']
        emitter_locs = dataset['emitter_locations']
        
        # All attack samples should have emitter locations
        for i, label in enumerate(labels):
            if label == 1:  # Attack
                assert emitter_locs[i] is not None
                assert len(emitter_locs[i]) == 3  # [x, y, z]
            else:  # Benign
                assert emitter_locs[i] is None
    
    def test_satellite_receptions(self, isac_system):
        """Test that satellite receptions are correctly generated."""
        num_sats = 12
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=10,
            num_satellites=num_sats,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )
        
        sat_recepts = dataset['satellite_receptions']
        
        for sample_sats in sat_recepts:
            assert len(sample_sats) == num_sats
            
            for sat in sample_sats:
                assert 'satellite_id' in sat
                assert 'position' in sat
                assert 'rx_time_padded' in sat
                assert 'distance' in sat
                assert len(sat['position']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])