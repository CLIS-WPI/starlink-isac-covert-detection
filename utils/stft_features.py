# ======================================
# 📄 utils/stft_features.py
# Purpose: STFT feature extraction for STNN-based localization
# Based on: Paper uses STFT to maintain consistent resolution (256x256)
# ======================================

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def compute_stft_cross_spectrum(signal1: np.ndarray, 
                                  signal2: np.ndarray,
                                  fs: float = 38400,
                                  nperseg: int = 256,
                                  noverlap: Optional[int] = None,
                                  output_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Compute cross-spectrum STFT for TDOA/FDOA estimation.
    
    This creates the input feature for STNN models.
    Paper uses STFT (not wavelet) to maintain consistent resolution.
    
    Args:
        signal1: First signal (e.g., satellite reception)
        signal2: Second signal (e.g., reference transmit signal)
        fs: Sampling frequency (Hz)
        nperseg: Length of each segment for STFT
        noverlap: Number of points to overlap between segments
        output_shape: Desired output shape (H, W)
    
    Returns:
        STFT cross-spectrum magnitude (H, W) normalized to [0, 1]
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Ensure signals are same length
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    # Compute STFT for both signals
    f1, t1, Zxx1 = signal.stft(signal1, fs=fs, nperseg=nperseg, 
                                 noverlap=noverlap, return_onesided=False)
    f2, t2, Zxx2 = signal.stft(signal2, fs=fs, nperseg=nperseg, 
                                 noverlap=noverlap, return_onesided=False)
    
    # Cross-spectrum (similar to GCC-PHAT but in freq-time domain)
    # This emphasizes time-frequency correlation patterns
    cross_spec = Zxx1 * np.conj(Zxx2)
    
    # Magnitude (phase information is in the complex values)
    magnitude = np.abs(cross_spec)
    
    # Normalize to [0, 1]
    magnitude = magnitude / (np.max(magnitude) + 1e-12)
    
    # Resize to target shape (256x256 as per paper)
    magnitude_resized = resize_stft(magnitude, output_shape)
    
    return magnitude_resized


def resize_stft(stft_matrix: np.ndarray, 
                target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize STFT matrix to target shape using interpolation.
    
    Args:
        stft_matrix: Input STFT matrix (H, W)
        target_shape: Target shape (H_target, W_target)
    
    Returns:
        Resized matrix
    """
    from scipy.ndimage import zoom
    
    zoom_factors = (target_shape[0] / stft_matrix.shape[0],
                    target_shape[1] / stft_matrix.shape[1])
    
    resized = zoom(stft_matrix, zoom_factors, order=1)  # Bilinear interpolation
    
    return resized


def compute_gcc_phat_stft(signal1: np.ndarray,
                           signal2: np.ndarray,
                           fs: float = 38400,
                           nperseg: int = 256,
                           output_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Compute GCC-PHAT style STFT (emphasizes phase correlation).
    
    Alternative feature representation that may work better for some scenarios.
    
    Args:
        signal1: First signal
        signal2: Second signal
        fs: Sampling frequency
        nperseg: STFT segment length
        output_shape: Output shape
    
    Returns:
        GCC-PHAT STFT magnitude (H, W)
    """
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    # STFT
    f1, t1, Zxx1 = signal.stft(signal1, fs=fs, nperseg=nperseg, 
                                 noverlap=nperseg//2, return_onesided=False)
    f2, t2, Zxx2 = signal.stft(signal2, fs=fs, nperseg=nperseg, 
                                 noverlap=nperseg//2, return_onesided=False)
    
    # GCC-PHAT: Normalize by magnitude
    cross_spec = Zxx1 * np.conj(Zxx2)
    magnitude = np.abs(cross_spec) + 1e-12
    gcc_phat = cross_spec / magnitude
    
    # Take magnitude of normalized cross-spectrum
    gcc_phat_mag = np.abs(gcc_phat)
    
    # Resize
    gcc_phat_resized = resize_stft(gcc_phat_mag, output_shape)
    
    return gcc_phat_resized


def compute_stft_features_batch(signals: np.ndarray,
                                  references: np.ndarray,
                                  fs: float = 38400,
                                  output_shape: Tuple[int, int] = (256, 256),
                                  method: str = 'cross_spectrum') -> np.ndarray:
    """
    Compute STFT features for a batch of signal pairs.
    
    Args:
        signals: Array of signals [N, T]
        references: Array of reference signals [N, T]
        fs: Sampling frequency
        output_shape: Output shape for each feature
        method: 'cross_spectrum' or 'gcc_phat'
    
    Returns:
        STFT features [N, H, W, 1]
    """
    N = len(signals)
    features = np.zeros((N, *output_shape, 1), dtype=np.float32)
    
    compute_fn = (compute_stft_cross_spectrum if method == 'cross_spectrum' 
                  else compute_gcc_phat_stft)
    
    for i in range(N):
        try:
            feat = compute_fn(signals[i], references[i], fs, 
                             nperseg=256, output_shape=output_shape)
            features[i, :, :, 0] = feat
        except Exception as e:
            print(f"Warning: Failed to compute STFT for sample {i}: {e}")
            features[i, :, :, 0] = 0.0  # Zero padding on failure
    
    return features


def visualize_stft_feature(stft_feature: np.ndarray, 
                            title: str = "STFT Feature",
                            save_path: Optional[str] = None):
    """
    Visualize STFT feature for debugging.
    
    Args:
        stft_feature: STFT feature (H, W) or (H, W, 1)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    
    if stft_feature.ndim == 3:
        stft_feature = stft_feature[:, :, 0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(stft_feature, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Normalized Magnitude')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def extract_stft_features_from_dataset(dataset: dict,
                                         fs: float = 38400,
                                         output_shape: Tuple[int, int] = (256, 256),
                                         max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract STFT features from dataset for STNN training.
    
    Args:
        dataset: Dataset dictionary with 'satellite_receptions' and 'tx_time_padded'
        fs: Sampling frequency
        output_shape: Output shape for features
        max_samples: Maximum number of samples to process (for testing)
    
    Returns:
        (stft_features, tdoa_labels, fdoa_labels)
        - stft_features: [N, H, W, 1]
        - tdoa_labels: [N] (seconds)
        - fdoa_labels: [N] (Hz)
    """
    sat_recepts = dataset.get('satellite_receptions', [])
    tx_padded = dataset.get('tx_time_padded', [])
    
    if len(sat_recepts) == 0 or len(tx_padded) == 0:
        raise ValueError("Dataset must contain 'satellite_receptions' and 'tx_time_padded'")
    
    features_list = []
    tdoa_labels_list = []
    fdoa_labels_list = []
    
    num_samples = len(sat_recepts) if max_samples is None else min(max_samples, len(sat_recepts))
    
    print(f"[STFT] Extracting features from {num_samples} samples...")
    
    for i in range(num_samples):
        sat_list = sat_recepts[i]
        ref_sig = tx_padded[i]
        
        if sat_list is None or len(sat_list) == 0:
            continue
        
        for sat in sat_list:
            try:
                # Get received signal (Path B for clean geometric delay)
                rx_sig = sat.get('rx_time_b_full', sat.get('rx_time_padded', None))
                
                if rx_sig is None:
                    continue
                
                # Compute STFT feature
                stft_feat = compute_stft_cross_spectrum(
                    rx_sig, ref_sig, fs=fs, output_shape=output_shape
                )
                
                # Get true TDOA (from geometry)
                true_delay_samples = sat.get('true_delay_samples', 0)
                tdoa_seconds = true_delay_samples / fs
                
                # Get true FDOA (if available)
                # Calculate from velocity and position
                c0 = 3e8
                sat_pos = np.array(sat['position'])
                sat_vel = np.array(sat.get('velocity', [0, 0, 0]))
                
                # Emitter location (from dataset or assume ground user)
                emitter_loc = dataset.get('emitter_locations', [None] * len(sat_recepts))[i]
                if emitter_loc is None:
                    emitter_loc = np.array([50e3, 50e3, 0.0])  # Default user location
                else:
                    emitter_loc = np.array(emitter_loc)
                
                # FDOA calculation (Doppler shift)
                fc = dataset.get('sampling_rate', fs)  # Carrier frequency approximation
                range_vec = emitter_loc - sat_pos
                range_norm = np.linalg.norm(range_vec) + 1e-9
                radial_velocity = np.dot(sat_vel, range_vec) / range_norm
                fdoa_hz = -(fc / c0) * radial_velocity  # Negative for approaching
                
                # Store
                features_list.append(stft_feat)
                tdoa_labels_list.append(tdoa_seconds)
                fdoa_labels_list.append(fdoa_hz)
                
            except Exception as e:
                print(f"Warning: Failed to process sample {i}, sat: {e}")
                continue
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")
    
    # Convert to arrays
    stft_features = np.array(features_list, dtype=np.float32)
    stft_features = stft_features[..., np.newaxis]  # Add channel dimension
    
    tdoa_labels = np.array(tdoa_labels_list, dtype=np.float32)
    fdoa_labels = np.array(fdoa_labels_list, dtype=np.float32)
    
    print(f"\n✓ Extracted {len(stft_features)} feature samples")
    print(f"  STFT shape: {stft_features.shape}")
    print(f"  TDOA range: [{tdoa_labels.min():.6f}, {tdoa_labels.max():.6f}] s")
    print(f"  FDOA range: [{fdoa_labels.min():.2f}, {fdoa_labels.max():.2f}] Hz")
    
    return stft_features, tdoa_labels, fdoa_labels


if __name__ == "__main__":
    # Test STFT feature extraction
    print("="*60)
    print("STFT Feature Extraction Test")
    print("="*60)
    
    # Generate dummy signals
    fs = 38400
    t = np.arange(0, 0.01, 1/fs)  # 10ms signal
    
    # Signal 1: Chirp
    signal1 = signal.chirp(t, f0=1000, f1=5000, t1=0.01, method='linear')
    
    # Signal 2: Delayed chirp (simulating TDOA)
    delay_samples = 50
    signal2 = np.pad(signal1[:-delay_samples], (delay_samples, 0), mode='constant')
    
    # Add noise
    signal1 += 0.1 * np.random.randn(len(signal1))
    signal2 += 0.1 * np.random.randn(len(signal2))
    
    # Compute STFT features
    print("\n[Test 1] Cross-spectrum STFT")
    stft_cross = compute_stft_cross_spectrum(signal1, signal2, fs=fs)
    print(f"✓ Shape: {stft_cross.shape}")
    print(f"✓ Range: [{stft_cross.min():.4f}, {stft_cross.max():.4f}]")
    
    print("\n[Test 2] GCC-PHAT STFT")
    stft_gcc = compute_gcc_phat_stft(signal1, signal2, fs=fs)
    print(f"✓ Shape: {stft_gcc.shape}")
    print(f"✓ Range: [{stft_gcc.min():.4f}, {stft_gcc.max():.4f}]")
    
    # Visualize
    try:
        visualize_stft_feature(stft_cross, "Cross-Spectrum STFT", 
                              save_path="/home/claude/stft_test_cross.png")
        visualize_stft_feature(stft_gcc, "GCC-PHAT STFT", 
                              save_path="/home/claude/stft_test_gcc.png")
        print("\n✓ Visualizations saved")
    except Exception as e:
        print(f"\n⚠️  Visualization failed: {e}")
    
    print("\n✓ STFT feature extraction test complete!")