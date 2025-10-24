"""
Test cases for detector model
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.detector import build_dual_input_cnn_h100, find_optimal_temperature
import tensorflow as tf


class TestDetector:
    
    def test_model_architecture(self):
        """Test that model has correct architecture."""
        model = build_dual_input_cnn_h100()
        
        # Check inputs
        assert len(model.inputs) == 2
        assert model.inputs[0].shape[1:] == (64, 64, 1)  # Spectrogram
        assert model.inputs[1].shape[1:] == (8, 8, 3)    # RX features
        
        # Check output
        assert model.output.shape[1] == 1  # Single logit output
        
    def test_model_compilation(self):
        """Test that model compiles without errors."""
        model = build_dual_input_cnn_h100()
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
        
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = build_dual_input_cnn_h100()
        
        # Create dummy inputs
        spec_input = np.random.randn(4, 64, 64, 1).astype(np.float32)
        rx_input = np.random.randn(4, 8, 8, 3).astype(np.float32)
        
        # Forward pass
        output = model([spec_input, rx_input], training=False)
        
        # Check output shape
        assert output.shape == (4, 1)
        
    def test_temperature_scaling(self):
        """Test temperature scaling calibration."""
        # Create simple model
        model = build_dual_input_cnn_h100()
        
        # Create dummy validation data
        Xs_val = np.random.randn(50, 64, 64, 1).astype(np.float32)
        Xr_val = np.random.randn(50, 8, 8, 3).astype(np.float32)
        y_val = np.random.randint(0, 2, 50).astype(np.float32)
        
        # Find optimal temperature
        temperature = find_optimal_temperature(
            model, Xs_val, Xr_val, y_val, max_iter=20
        )
        
        # Temperature should be positive
        assert temperature > 0
        assert temperature < 10  # Reasonable range
        
    def test_prediction_shape(self):
        """Test that predictions have correct shape."""
        model = build_dual_input_cnn_h100()
        
        # Create dummy test data
        Xs_te = np.random.randn(10, 64, 64, 1).astype(np.float32)
        Xr_te = np.random.randn(10, 8, 8, 3).astype(np.float32)
        
        # Get predictions
        logits = model.predict([Xs_te, Xr_te], verbose=0)
        
        # Check shape
        assert logits.shape == (10, 1)
        
        # Apply sigmoid
        probs = tf.nn.sigmoid(logits).numpy()
        
        # Probabilities should be in [0, 1]
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])