"""
Quick test runner without external dependencies
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from core.isac_system import ISACSystem
        from core.dataset_generator import generate_dataset_multi_satellite
        from core.feature_extraction import extract_features_and_split
        from core.localization import gcc_phat, trilateration_2d_wls
        from core.covert_injection import inject_covert_channel
        from model.detector import build_dual_input_cnn_h100
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from model.detector import build_dual_input_cnn_h100
        model = build_dual_input_cnn_h100()
        print(f"✓ Model created with {model.count_params():,} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_isac_system():
    """Test ISAC system initialization."""
    print("\nTesting ISAC system...")
    try:
        from core.isac_system import ISACSystem
        isac = ISACSystem()
        print(f"✓ ISAC system initialized")
        print(f"  - Sampling rate: {isac.SAMPLING_RATE/1e6:.1f} MHz")
        print(f"  - FFT size: {isac.FFT_SIZE}")
        print(f"  - OFDM symbols: {isac.NUM_OFDM_SYMBOLS}")
        return True
    except Exception as e:
        print(f"✗ ISAC system failed: {e}")
        return False

def test_gcc_phat():
    """Test GCC-PHAT correlation."""
    print("\nTesting GCC-PHAT...")
    try:
        import numpy as np
        from core.localization import gcc_phat
        
        # Create test signals
        fs = 10e6
        t = np.arange(0, 0.001, 1/fs)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        
        # Add delay
        delayed = np.roll(signal, 100)
        
        # Correlate
        corr = gcc_phat(signal, delayed, upsample_factor=32)
        
        # Find peak
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        delay_est = (peak_idx - center) / 32
        
        print(f"✓ GCC-PHAT working")
        print(f"  - True delay: 100 samples")
        print(f"  - Estimated: {delay_est:.1f} samples")
        print(f"  - Error: {abs(delay_est - 100):.1f} samples")
        
        return abs(delay_est - 100) < 5
    except Exception as e:
        print(f"✗ GCC-PHAT failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    try:
        import numpy as np
        from core.feature_extraction import extract_spectrogram_tf
        
        # Create dummy data
        iq_samples = np.random.randn(5, 720).astype(np.complex64)
        
        # Extract features
        specs = extract_spectrogram_tf(iq_samples)
        
        print(f"✓ Feature extraction working")
        print(f"  - Input shape: {iq_samples.shape}")
        print(f"  - Output shape: {specs.shape}")
        
        return specs.shape == (5, 64, 64, 1)
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("="*60)
    print("QUICK TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_isac_system,
        test_gcc_phat,
        test_feature_extraction
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1
    print("="*60)

if __name__ == "__main__":
    sys.exit(main())