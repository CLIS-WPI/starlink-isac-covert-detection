"""
Test cases for TDoA localization
"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.localization import gcc_phat, trilateration_2d_wls


class TestLocalization:
    
    def test_gcc_phat_delay(self):
        """Test GCC-PHAT with known delay."""
        # Create reference signal
        fs = 10e6  # 10 MHz sampling rate
        t = np.arange(0, 0.001, 1/fs)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)  # 1 MHz carrier
        
        # Add known delay (100 samples)
        true_delay = 100
        delayed_signal = np.roll(signal, true_delay)
        
        # GCC-PHAT
        corr = gcc_phat(signal, delayed_signal, upsample_factor=32)
        
        # Find peak
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        estimated_delay = (peak_idx - center) / 32
        
        # Should be close to true delay
        assert abs(estimated_delay - true_delay) < 5  # Within 5 samples
        
    def test_trilateration_2d(self):
        """Test 2D trilateration with known position."""
        # Known emitter position
        true_pos = np.array([1000, 2000, 0])
        
        # Satellite positions (square around emitter)
        sat_positions = [
            np.array([0, 0, 600e3]),
            np.array([5000, 0, 600e3]),
            np.array([0, 5000, 600e3]),
            np.array([5000, 5000, 600e3])
        ]
        
        # Reference satellite (first one)
        ref_pos = sat_positions[0]
        
        # Calculate true TDoA differences
        c0 = 3e8
        d_ref = np.linalg.norm(true_pos - ref_pos)
        
        tdoa_diffs = []
        pairs = []
        weights = []
        
        for sat_pos in sat_positions[1:]:
            d_i = np.linalg.norm(true_pos - sat_pos)
            tdoa_diff = d_i - d_ref
            
            tdoa_diffs.append(tdoa_diff)
            pairs.append((ref_pos, sat_pos))
            weights.append(1.0)
        
        # Estimate position
        estimated_pos = trilateration_2d_wls(tdoa_diffs, pairs, weights)
        
        # Error should be small
        error = np.linalg.norm(estimated_pos[:2] - true_pos[:2])
        assert error < 100  # Within 100 meters
        
    def test_gcc_phat_noise_robustness(self):
        """Test GCC-PHAT robustness to noise."""
        # Create clean signal
        fs = 10e6
        t = np.arange(0, 0.001, 1/fs)
        signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        
        # Add delay
        true_delay = 50
        delayed_signal = np.roll(signal, true_delay)
        
        # Add AWGN (SNR = 10 dB)
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / 10
        noise = (np.random.randn(len(signal)) + 
                1j * np.random.randn(len(signal))) * np.sqrt(noise_power/2)
        
        noisy_delayed = delayed_signal + noise
        
        # GCC-PHAT with beta=0.8 for better noise performance
        corr = gcc_phat(signal, noisy_delayed, upsample_factor=32, beta=0.8)
        
        # Find peak
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        estimated_delay = (peak_idx - center) / 32
        
        # Should still be close despite noise
        assert abs(estimated_delay - true_delay) < 10  # Within 10 samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])