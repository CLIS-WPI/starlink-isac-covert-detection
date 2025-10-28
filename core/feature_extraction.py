# ======================================
# core/feature_extraction.py
# Purpose: GPU-optimized feature extraction (spectrogram + RX features + STFT per satellite)
# ======================================

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GroupShuffleSplit


@tf.function  # Enable graph mode for better GPU utilization
def extract_spectrogram_tf(iq_batch, n_fft=128, frame_length=128, 
                           frame_step=32, out_hw=(64, 64)):
    """
    GPU-accelerated spectrogram from IQ samples [B, T]
    """
    iq_batch = tf.convert_to_tensor(iq_batch, dtype=tf.complex64)
    x_mag = tf.abs(iq_batch)
    
    stft_c = tf.signal.stft(
        x_mag,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=n_fft
    )
    
    spec = tf.abs(stft_c)
    spec = tf.expand_dims(spec, axis=-1)
    spec = tf.image.resize(spec, out_hw)
    spec = spec / (tf.reduce_max(spec, axis=[1, 2, 3], keepdims=True) + 1e-8)
    
    return tf.cast(spec, tf.float32)


@tf.function  # Enable graph mode for better GPU utilization
def extract_received_signal_features(dataset):
    """
    Per-subcarrier statistics from CSI [B, SYM, SC]
    """
    csi = tf.convert_to_tensor(dataset['csi'], dtype=tf.complex64)
    pwr = tf.abs(csi) ** 2
    pwr = pwr / (tf.reduce_max(pwr, axis=[1, 2], keepdims=True) + 1e-12)
    
    mean_sc = tf.reduce_mean(pwr, axis=1)
    std_sc = tf.math.reduce_std(pwr, axis=1)
    max_sc = tf.reduce_max(pwr, axis=1)
    
    F = tf.stack([mean_sc, std_sc, max_sc], axis=-1)
    
    num_sc = tf.shape(F)[1]
    def trim(): return F[:, :64, :]
    def pad():  return tf.pad(F, [[0, 0], [0, 64 - num_sc], [0, 0]])
    F = tf.cond(num_sc >= 64, trim, pad)
    
    F = tf.reshape(F, [-1, 8, 8, 3])
    return tf.cast(F, tf.float32)


@tf.function  # Enable graph mode for better GPU utilization
def extract_stft_per_satellite(sat_receptions, sampling_rate, 
                               frame_length=64, frame_step=32, out_hw=(32, 32)):
    """
    NEW: STFT from rx_time of each satellite (for STNN-aid CAF)
    Input: sat_receptions list of dicts with 'rx_time' [T]
    Output: [B, num_sat, H, W, 1]
    """
    
    batch_size = len(sat_receptions)
    num_sat = len(sat_receptions[0]) if batch_size > 0 else 0
    features = []

    for b in range(batch_size):
        sat_features = []
        for sat in sat_receptions[b]:
            signal = tf.convert_to_tensor(sat['rx_time'], dtype=tf.complex64)
            signal = tf.abs(signal)  # magnitude

            stft = tf.signal.stft(
                signal,
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=frame_length
            )
            mag = tf.abs(stft)
            mag = tf.expand_dims(mag, -1)
            mag = tf.image.resize(mag, out_hw)
            mag = mag / (tf.reduce_max(mag) + 1e-8)
            sat_features.append(mag)

        # [num_sat, H, W, 1]
        sat_features = tf.stack(sat_features)
        features.append(sat_features)

    # [B, num_sat, H, W, 1]
    return tf.stack(features)


def sat_group_id(sat_rx_list):
    """Group by satellite shell altitude"""
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
    Extract: 
      - Spectrogram from IQ
      - RX features from CSI
      - STFT per satellite (NEW)
    Optimized for GPU: Process all at once to minimize CPU-GPU transfers
    """
    print("\n[Feature] Extracting features (GPU)...")
    print("Note: Processing large batches on GPU for maximum efficiency...")
    
    total_samples = len(dataset['iq_samples'])
    
    # Use larger batch size for better GPU utilization
    batch_size = 256  # Larger batches = better GPU utilization
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"Processing {total_samples} samples in {num_batches} batches (batch_size={batch_size})...")
    
    feats_spec_list = []
    feats_rx_list = []
    feats_stft_list = []
    
    with tf.device('/GPU:0'):  # Explicitly force GPU execution
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            
            print(f"  Batch {i+1}/{num_batches} [{start_idx}:{end_idx}] - Processing on GPU...")
            
            # 1. Spectrogram from IQ (batch) - stays on GPU
            batch_spec = extract_spectrogram_tf(
                dataset['iq_samples'][start_idx:end_idx],
                n_fft=128, frame_length=128, frame_step=32, out_hw=(64, 64)
            )
            feats_spec_list.append(batch_spec)  # Keep as TensorFlow tensor
            
            # 2. RX features from CSI (batch) - stays on GPU
            batch_dataset = {
                'csi': dataset['csi'][start_idx:end_idx]
            }
            batch_rx = extract_received_signal_features(batch_dataset)
            feats_rx_list.append(batch_rx)  # Keep as TensorFlow tensor
            
            # 3. STFT per satellite (batch) - stays on GPU
            sat_recepts = dataset.get('satellite_receptions', None)
            if sat_recepts is not None:
                batch_sat_recepts = sat_recepts[start_idx:end_idx]
                batch_stft = extract_stft_per_satellite(
                    batch_sat_recepts,
                    sampling_rate=dataset['sampling_rate'],
                    frame_length=64, frame_step=32, out_hw=(32, 32)
                )
                feats_stft_list.append(batch_stft)  # Keep as TensorFlow tensor
    
    print("✓ GPU processing complete. Transferring results to CPU...")
    
    # NOW transfer everything to CPU in one go (efficient)
    feats_spec = tf.concat(feats_spec_list, axis=0).numpy()
    feats_rx = tf.concat(feats_rx_list, axis=0).numpy()
    
    if sat_recepts is not None:
        feats_stft_sat = tf.concat(feats_stft_list, axis=0).numpy()
    else:
        feats_stft_sat = None

    # Convert to NumPy
    feats_spec = np.array(feats_spec)
    feats_rx = np.array(feats_rx)
    labels = np.array(dataset['labels']).astype(np.float32)
    
    # Align
    m = min(len(feats_spec), len(feats_rx), len(labels))
    feats_spec, feats_rx, labels = feats_spec[:m], feats_rx[:m], labels[:m]
    if feats_stft_sat is not None:
        feats_stft_sat = np.array(feats_stft_sat)[:m]

    # Groups
    print("[Feature] Creating constellation-based groups...")
    if sat_recepts is None:
        groups = np.array([hash(None)] * m)
    else:
        groups = np.array([sat_group_id(sr) for sr in sat_recepts[:m]])
    
    unique_groups = len(np.unique(groups))
    print(f"Found {unique_groups} unique satellite constellations")

    # Split
    if unique_groups < 2:
        print("Only one group — using stratified split")
        Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            feats_spec, feats_rx, labels, np.arange(m),
            test_size=0.2, random_state=42, stratify=labels
        )
        if feats_stft_sat is not None:
            Xstft_tr, Xstft_te = feats_stft_sat[idx_tr], feats_stft_sat[idx_te]
        else:
            Xstft_tr, Xstft_te = None, None
    else:
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(np.arange(len(groups)), groups=groups))
        
        Xs_tr, Xs_te = feats_spec[train_idx], feats_spec[test_idx]
        Xr_tr, Xr_te = feats_rx[train_idx], feats_rx[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        idx_tr, idx_te = train_idx, test_idx
        
        if feats_stft_sat is not None:
            Xstft_tr, Xstft_te = feats_stft_sat[train_idx], feats_stft_sat[test_idx]
        else:
            Xstft_tr, Xstft_te = None, None

        print(f"Train groups: {len(np.unique(groups[train_idx]))}")
        print(f"Test groups: {len(np.unique(groups[test_idx]))}")
        if set(groups[train_idx]) & set(groups[test_idx]):
            print("WARNING: Group overlap!")
        else:
            print("No group overlap — proper split!")

    print(f"Train: {len(Xs_tr)} samples")
    print(f"Test: {len(Xs_te)} samples")

    return (Xs_tr, Xs_te, Xr_tr, Xr_te, Xstft_tr, Xstft_te, 
            y_tr, y_te, idx_tr, idx_te)