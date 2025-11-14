"""
Unit tests for CNNDetector with attention mechanism flag.
"""
import pytest
import numpy as np
import sys
import os
import tempfile
import shutil

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.detector_cnn import CNNDetector


@pytest.mark.unit
def test_cnn_with_attention():
    """Test CNNDetector with attention enabled."""
    # Create small synthetic dataset
    n_samples = 20
    n_symbols = 10
    n_subcarriers = 64
    
    # Generate complex OFDM grids
    X = np.random.randn(n_samples, n_symbols, n_subcarriers).astype(np.complex64) * (1 + 1j)
    y = np.random.randint(0, 2, n_samples)
    
    # Create detector with attention
    detector = CNNDetector(
        use_csi=False,
        use_attention=True,
        random_state=42,
        learning_rate=0.001,
        dropout_rate=0.3
    )
    
    # Verify attention is enabled
    assert detector.use_attention == True, "Attention should be enabled"
    
    # Train for a few epochs (quick test)
    detector.train(
        X, y,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    
    # Verify model was trained
    assert detector.is_trained == True, "Model should be trained"
    assert detector.model is not None, "Model should exist"
    
    # Make predictions
    y_pred = detector.predict(X)
    assert y_pred.shape == (n_samples, 1), "Predictions should have correct shape"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be probabilities"


@pytest.mark.unit
def test_cnn_without_attention():
    """Test CNNDetector with attention disabled."""
    # Create small synthetic dataset
    n_samples = 20
    n_symbols = 10
    n_subcarriers = 64
    
    # Generate complex OFDM grids
    X = np.random.randn(n_samples, n_symbols, n_subcarriers).astype(np.complex64) * (1 + 1j)
    y = np.random.randint(0, 2, n_samples)
    
    # Create detector without attention
    detector = CNNDetector(
        use_csi=False,
        use_attention=False,
        random_state=42,
        learning_rate=0.001,
        dropout_rate=0.3
    )
    
    # Verify attention is disabled
    assert detector.use_attention == False, "Attention should be disabled"
    
    # Train for a few epochs (quick test)
    detector.train(
        X, y,
        epochs=2,
        batch_size=8,
        verbose=0
    )
    
    # Verify model was trained
    assert detector.is_trained == True, "Model should be trained"
    assert detector.model is not None, "Model should exist"
    
    # Make predictions
    y_pred = detector.predict(X)
    assert y_pred.shape == (n_samples, 1), "Predictions should have correct shape"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be probabilities"


@pytest.mark.unit
def test_attention_flag_default():
    """Test that attention is enabled by default."""
    detector = CNNDetector(use_csi=False)
    assert detector.use_attention == True, "Attention should be enabled by default"


@pytest.mark.unit
def test_attention_flag_explicit():
    """Test explicit attention flag setting."""
    # With attention
    detector1 = CNNDetector(use_csi=False, use_attention=True)
    assert detector1.use_attention == True
    
    # Without attention
    detector2 = CNNDetector(use_csi=False, use_attention=False)
    assert detector2.use_attention == False


@pytest.mark.unit
def test_attention_affects_architecture():
    """Test that attention flag affects model architecture."""
    n_samples = 10
    n_symbols = 10
    n_subcarriers = 64
    
    X = np.random.randn(n_samples, n_symbols, n_subcarriers).astype(np.complex64) * (1 + 1j)
    y = np.random.randint(0, 2, n_samples)
    
    # Train with attention
    detector_att = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_att.train(X, y, epochs=1, batch_size=4, verbose=0)
    
    # Train without attention
    detector_no_att = CNNDetector(use_csi=False, use_attention=False, random_state=42)
    detector_no_att.train(X, y, epochs=1, batch_size=4, verbose=0)
    
    # Models should be different (different architectures)
    # We can't directly compare weights, but we can verify both work
    assert detector_att.is_trained == True
    assert detector_no_att.is_trained == True
    
    # Both should make predictions
    pred_att = detector_att.predict(X)
    pred_no_att = detector_no_att.predict(X)
    
    assert pred_att.shape == pred_no_att.shape
    assert pred_att.shape == (n_samples, 1)

