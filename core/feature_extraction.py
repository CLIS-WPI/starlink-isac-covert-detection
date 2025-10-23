# ======================================
# 📄 core/feature_extraction.py
# Purpose: GPU-optimized feature extraction (spectrogram + RX features)
# ======================================

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit


@tf.function(jit_compile=True)
def extract_spectrogram_tf(iq_batch, n_fft=128, frame_length=128, 
                           frame_step=32, out_hw=(64, 64)):
    """
    GPU-accelerated spectrogram extraction using TensorFlow STFT.
    
    Args:
        iq_batch: Complex IQ samples [B, T]
        n_fft: FFT size
        frame_length: STFT frame length
        frame_step: STFT hop size
        out_hw: Output height and width
    
    Returns:
        Tensor: Spectrograms [B, H, W, 1]
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    iq_batch = tf.convert_to_tensor(iq_batch, dtype=tf.complex64)
    x_mag = tf.abs(iq_batch)
    
    # Compute STFT on GPU
    stft_c = tf.signal.stft(
        x_mag,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=n_fft
    )
    
    # Magnitude spectrogram
    spec = tf.abs(stft_c)
    spec = tf.expand_dims(spec, axis=-1)
    
    H, W = out_hw
    spec = tf.image.resize(spec, [H, W])
    
    # Normalize
    spec = spec / (tf.reduce_max(spec, axis=[1, 2, 3], keepdims=True) + 1e-8)
    
    return tf.cast(spec, tf.float32)


@tf.function(jit_compile=True)
def extract_received_signal_features(dataset):
    """
    GPU-optimized RX feature extraction (per-subcarrier statistics).
    
    Args:
        dataset: Dictionary with 'csi' field [B, SYM, SC]
    
    Returns:
        Tensor: RX features [B, 8, 8, 3]
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    csi = tf.convert_to_tensor(dataset['csi'], dtype=tf.complex64)
    pwr = tf.abs(csi) ** 2
    pwr = pwr / (tf.reduce_max(pwr, axis=[1, 2], keepdims=True) + 1e-12)
    
    # Statistics across OFDM symbols
    mean_sc = tf.reduce_mean(pwr, axis=1)
    std_sc = tf.math.reduce_std(pwr, axis=1)
    max_sc = tf.reduce_max(pwr, axis=1)
    
    # Stack features
    F = tf.stack([mean_sc, std_sc, max_sc], axis=-1)
    
    # Trim/pad to 64 subcarriers
    num_sc = tf.shape(F)[1]
    def trim():
        return F[:, :64, :]
    def pad():
        pad_len = 64 - num_sc
        return tf.pad(F, [[0, 0], [0, pad_len], [0, 0]])
    F = tf.cond(num_sc >= 64, trim, pad)
    
    # Reshape to 8×8×3
    F = tf.reshape(F, [-1, 8, 8, 3])
    
    return tf.cast(F, tf.float32)


def sat_group_id(sat_rx_list):
    """
    Create group ID based on satellite constellation geometry.
    Groups samples with same altitude shells together.
    """
    if sat_rx_list is None:
        return hash(None)
    try:
        def get_shell(alt_km):
            if 300 <= alt_km < 400: return 3
            if 400 <= alt_km < 500: return 4
            if 500 <= alt_km < 600: return 5
            return 6
        
        shells = [get_shell(s['position'][2]/1000) for s in sat_rx_list]
        shell_counts = tuple(sorted(np.bincount(shells)))
        return hash((shell_counts, len(sat_rx_list)))
    except Exception:
        return hash(str(sat_rx_list))


def extract_features_and_split(dataset):
    """
    Extract features and perform group-based train/test split.
    
    Args:
        dataset: Dictionary from dataset_generator
    
    Returns:
        tuple: (Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te)
    """
    print("\n[Feature] Extracting features (GPU)...")
    
    # Extract features
    feats_spec = extract_spectrogram_tf(
        dataset['iq_samples'],
        n_fft=128,
        frame_length=128,
        frame_step=32,
        out_hw=(64, 64)
    )
    feats_rx = extract_received_signal_features(dataset)
    
    # Convert to NumPy
    feats_spec = np.array(feats_spec)
    feats_rx = np.array(feats_rx)
    labels = np.array(dataset['labels']).astype(np.float32)
    
    # Align sizes
    m = min(len(feats_spec), len(feats_rx), len(labels))
    feats_spec, feats_rx, labels = feats_spec[:m], feats_rx[:m], labels[:m]
    
    # Create constellation-based groups
    print("[Feature] Creating constellation-based groups...")
    sat_recepts = dataset.get('satellite_receptions', None)
    if sat_recepts is None:
        groups = np.array([hash(None)] * m)
    else:
        groups = np.array([sat_group_id(sr) for sr in sat_recepts[:m]])
    
    unique_groups = len(np.unique(groups))
    print(f"✓ Found {unique_groups} unique satellite constellations")
    
    # Group-based split
    if unique_groups < 2:
        print("⚠️ Only one group — using stratified random split")
        Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            feats_spec, feats_rx, labels, np.arange(m),
            test_size=0.2, random_state=42, stratify=labels
        )
    else:
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(np.arange(len(groups)), groups=groups))
        
        Xs_tr, Xs_te = feats_spec[train_idx], feats_spec[test_idx]
        Xr_tr, Xr_te = feats_rx[train_idx], feats_rx[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        idx_tr, idx_te = train_idx, test_idx
        
        print(f"✓ Train groups: {len(np.unique(groups[train_idx]))}")
        print(f"✓ Test groups: {len(np.unique(groups[test_idx]))}")
        
        # Verify no overlap
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            print(f"⚠️ WARNING: {len(overlap)} groups overlap!")
        else:
            print("✓ No group overlap — proper split! 🎯")
    
    print(f"✓ Train: {len(Xs_tr)} samples")
    print(f"✓ Test: {len(Xs_te)} samples")
    
    return Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te