"""
Enhanced Test Cases for Dataset Generation
Purpose: Physical correctness, power integrity, and consistency of ISAC dataset
"""
import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.isac_system import ISACSystem
from core.dataset_generator import generate_dataset_multi_satellite


class TestDatasetGenerator:
    
    @pytest.fixture(scope="module")
    def isac_system(self):
        """Create ISAC system once for all tests."""
        return ISACSystem()
    
    # ==============================================================
    # ✅ BASIC STRUCTURE TEST
    # ==============================================================
    def test_dataset_structure(self, isac_system):
        """Dataset should have consistent and complete structure."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=5,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )
        
        keys = ['iq_samples', 'csi', 'labels', 'emitter_locations', 'satellite_receptions']
        for key in keys:
            assert key in dataset, f"Missing key: {key}"
        
        n = len(dataset['labels'])
        assert dataset['iq_samples'].shape[0] == n
        assert dataset['csi'].shape[0] == n
        assert len(dataset['emitter_locations']) == n

    # ==============================================================
    # ✅ POWER RATIO TEST
    # ==============================================================
    def test_power_ratio_reasonable(self, isac_system):
        """Attack power should be higher than benign within expected ratio (~1.1–1.5×)."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=30,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )

        labels = np.array(dataset['labels'])
        benign_idx = np.where(labels == 0)[0]
        attack_idx = np.where(labels == 1)[0]

        benign_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in benign_idx])
        attack_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in attack_idx])
        power_ratio = attack_power / benign_power

        print(f"\n[TEST] Power ratio: {power_ratio:.3f}")
        assert 1.05 <= power_ratio <= 1.6, f"Unrealistic power ratio: {power_ratio:.3f}"

    # ==============================================================
    # ✅ EMITTER LOCATION TEST
    # ==============================================================
    def test_emitter_locations_validity(self, isac_system):
        """Every attack sample must have a valid (x,y,z) emitter location."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=10,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )

        for label, loc in zip(dataset['labels'], dataset['emitter_locations']):
            if label == 1:  # attack
                assert loc is not None, "Attack sample missing emitter location"
                assert len(loc) == 3, "Emitter location should have 3 coordinates"
                assert all(np.isfinite(loc)), "Invalid emitter coordinates (NaN/inf)"
            else:  # benign
                assert loc is None, "Benign sample should not have emitter location"

    # ==============================================================
    # ✅ TOPOLOGY GEOMETRY TEST
    # ==============================================================
    def test_satellite_geometry_consistency(self, isac_system):
        """Validate that all satellites have valid 3D positions and realistic distances."""
        num_sats = 6
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=5,
            num_satellites=num_sats,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )

        sat_recepts = dataset['satellite_receptions']
        for sample_sats in sat_recepts:
            assert len(sample_sats) == num_sats
            for sat in sample_sats:
                pos = np.array(sat['position'])
                assert pos.shape == (3,)
                dist = sat.get('distance', None)
                assert dist is None or dist > 100e3, f"Distance too small: {dist}"
                assert np.linalg.norm(pos) > 1e5, "Invalid satellite position (too close to origin)"

    # ==============================================================
    # ✅ PATH CONSISTENCY TEST (A vs B)
    # ==============================================================
    def test_path_consistency(self, isac_system):
        """Ensure that Path A (rx_time) and Path B (rx_time_padded_final) are synchronized in shape."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=5,
            num_satellites=3,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )

        sat = dataset['satellite_receptions'][0][0]
        rx_a = sat.get('rx_time', None)
        rx_b = sat.get('rx_time_padded', None)
        assert rx_a is not None and rx_b is not None, "Missing rx_time or rx_time_padded"
        assert rx_a.shape[-1] == rx_b.shape[-1], f"Mismatch in Path A/B length: {rx_a.shape} vs {rx_b.shape}"

        # Optional: simple correlation test
        corr = np.correlate(np.abs(rx_a), np.abs(rx_b), mode='valid')
        assert np.max(corr) > 0, "Zero correlation between Path A and Path B"

    # ==============================================================
    # ✅ NOISE LEVEL CHECK
    # ==============================================================
    def test_noise_floor_realism(self, isac_system):
        """Noise floor should not dominate or vanish."""
        dataset = generate_dataset_multi_satellite(
            isac_system,
            num_samples_per_class=10,
            num_satellites=4,
            ebno_db_range=(10, 15),
            covert_rate_mbps_range=(10, 30)
        )

        iq = dataset['iq_samples']
        noise_floor = np.mean(np.abs(iq)**2) / np.max(np.abs(iq)**2)
        print(f"\n[TEST] Noise floor ratio: {noise_floor:.6f}")
        assert 1e-4 < noise_floor < 1e-1, "Unrealistic noise floor range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
