"""
Test cases for covert channel injection
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.covert_injection import inject_covert_channel
from sionna.phy.ofdm import ResourceGrid


class TestCovertInjection:
    
    @pytest.fixture
    def resource_grid(self):
        """Create resource grid for testing."""
        return ResourceGrid(
            num_ofdm_symbols=10,
            fft_size=64,
            subcarrier_spacing=60e3,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=8,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 7]
        )
    
    def test_power_preservation(self, resource_grid):
        """Test that covert injection preserves power."""
        import tensorflow as tf
        
        # Create random OFDM frame
        ofdm_frame = tf.complex(
            tf.random.normal([1, 1, 1, 10, 64]),
            tf.random.normal([1, 1, 1, 10, 64])
        )
        
        # Measure original power
        orig_power = tf.reduce_mean(tf.abs(ofdm_frame)**2).numpy()
        
        # Inject covert channel
        injected_frame, emitter_loc = inject_covert_channel(
            ofdm_frame,
            resource_grid,
            covert_rate_mbps=20,
            scs=60e3,
            covert_amp=2.0
        )
        
        # Measure new power
        new_power = tf.reduce_mean(tf.abs(injected_frame)**2).numpy()
        
        # Power should be preserved (within 1%)
        power_ratio = new_power / orig_power
        assert 0.99 <= power_ratio <= 1.01, f"Power ratio: {power_ratio}"
        
    def test_emitter_location(self, resource_grid):
        """Test that emitter location is generated."""
        import tensorflow as tf
        
        ofdm_frame = tf.complex(
            tf.random.normal([1, 1, 1, 10, 64]),
            tf.random.normal([1, 1, 1, 10, 64])
        )
        
        _, emitter_loc = inject_covert_channel(
            ofdm_frame,
            resource_grid,
            covert_rate_mbps=20,
            scs=60e3,
            covert_amp=2.0
        )
        
        # Should have 3D location
        assert len(emitter_loc) == 3
        assert emitter_loc[2] == 0.0  # Ground level
        
    def test_no_injection_when_zero_rate(self, resource_grid):
        """Test that no injection happens when rate is 0."""
        import tensorflow as tf
        
        ofdm_frame = tf.complex(
            tf.random.normal([1, 1, 1, 10, 64]),
            tf.random.normal([1, 1, 1, 10, 64])
        )
        
        # Zero rate should return original frame
        result_frame, emitter_loc = inject_covert_channel(
            ofdm_frame,
            resource_grid,
            covert_rate_mbps=0,
            scs=60e3,
            covert_amp=2.0
        )
        
        # Should be unchanged
        np.testing.assert_array_equal(ofdm_frame.numpy(), result_frame.numpy())
        assert emitter_loc is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])