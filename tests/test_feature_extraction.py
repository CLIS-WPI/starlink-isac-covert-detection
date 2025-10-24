"""
Test cases for feature extraction
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.feature_extraction import extract_spectrogram_tf, extract_received_signal_features
import tensorflow as tf


class TestFeatureExtraction:
    
    def test_spectrogram_shape(self):
        """Test spectrogram extraction shape."""
        # Create dummy IQ samples
        batch_size = 10
        signal_length = 720
        iq_samples = np.random.randn(batch_size, signal_length).astype(np.complex64)
        
        # Extract spectrograms
        spectrograms = extract_spectrogram_tf(
            iq_samples,
            n_fft=128,
            frame_length=128,
            frame_step=32,
            out_hw=(64, 64)
        )
        
        # Check shape
        assert spectrograms.shape == (batch_size, 64, 64, 1)
        
        # Check normalization
        assert np.all(spectrograms >= 0)
        assert np.all(spectrograms <= 1)
        
    def test_rx_features_shape(self):
        """Test RX feature extraction shape."""
        # Create dummy dataset
        batch_size = 10
        num_symbols = 10
        num_subcarriers = 64
        
        csi = np.random.randn(batch_size, num_symbols, num_subcarriers).astype(np.complex64)
        
        dataset = {'csi': csi}
        
        # Extract features
        rx_features = extract_received_signal_features(dataset)
        
        # Check shape
        assert rx_features.shape == (batch_size, 8, 8, 3)
        
        # Check non-negative
        assert np.all(rx_features >= 0)
        
    def test_spectrogram_reproducibility(self):
        """Test that same input gives same output."""
        iq_samples = np.random.randn(5, 720).astype(np.complex64)
        
        spec1 = extract_spectrogram_tf(iq_samples)
        spec2 = extract_spectrogram_tf(iq_samples)
        
        # Should be identical
        np.testing.assert_array_almost_equal(spec1, spec2, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])