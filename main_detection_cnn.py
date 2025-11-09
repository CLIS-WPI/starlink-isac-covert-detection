#!/usr/bin/env python3
"""
🧠 CNN DETECTION PIPELINE
=========================
Main detection pipeline using CNN-based detector (with optional CSI fusion).

This replaces the RandomForest approach with deep learning for:
- Ultra-subtle covert channel detection (< 1% power difference)
- Automatic feature learning from raw OFDM grids
- Optional multi-modal fusion with CSI data

Usage:
    python3 main_detection_cnn.py --use-csi  # With CSI fusion
    python3 main_detection_cnn.py --batch-size 512  # H100 optimized
    python3 main_detection_cnn.py --multi-gpu  # Use all GPUs
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.signal import stft
from sklearn.model_selection import train_test_split

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# Configuration
from config.settings import (
    init_directories,
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA,
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR,
    GLOBAL_SEED,
    SEED,  # Alias for backward compatibility
    USE_SPECTROGRAM,
    USE_FOCAL_LOSS,
    FOCAL_LOSS_GAMMA,
    FOCAL_LOSS_ALPHA
)

# 🔒 Phase 0: Set global seeds for reproducibility
from utils.reproducibility import set_global_seeds, log_seed_info
log_seed_info("main_detection_cnn.py")
set_global_seeds(deterministic=True)

# CNN Detector
from model.detector_cnn import CNNDetector


def setup_gpu_strategy(multi_gpu=False):
    """
    Setup GPU strategy for training.
    
    Args:
        multi_gpu: Use all available GPUs (MirroredStrategy)
    
    Returns:
        strategy: TF distribution strategy
    """
    if multi_gpu:
        # Use all available GPUs
        strategy = tf.distribute.MirroredStrategy()
        print(f"  ✓ Multi-GPU training: {strategy.num_replicas_in_sync} GPUs")
    else:
        # Single GPU (default)
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print(f"  ✓ Single GPU training")
    
    return strategy
def compute_spectrogram(grids):
    """
    Convert OFDM grids to spectrograms using STFT.
    
    Args:
        grids: Complex OFDM grids (N, symbols, subcarriers, 1)
    
    Returns:
        spectrograms: (N, freq_bins, time_frames, channels)
    """
    if not USE_SPECTROGRAM:
        return grids
    
    print("  🔄 Computing spectrograms using STFT...")
    
    # Handle different input shapes
    grids = np.squeeze(grids)  # Remove all size-1 dimensions
    
    # Ensure 3D: (N, symbols, subcarriers)
    if grids.ndim == 2:
        grids = grids[np.newaxis, :, :]
    
    N, symbols, subcarriers = grids.shape
    
    # Flatten to (N, symbols * subcarriers) for STFT
    signals = grids.reshape(N, -1)
    
    # Convert complex to real by taking magnitude (IQ data)
    # Or concatenate real/imag: (N, 2 * len)
    signals_real = np.abs(signals).astype(np.float32)
    
    # Convert to TensorFlow tensors
    signals_tf = tf.constant(signals_real, dtype=tf.float32)
    
    # Apply STFT: frame_length=128, frame_step=64
    frame_length = 128
    frame_step = 64
    fft_length = 256
    
    spectrograms = stft(
        signals_tf,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )
    
    # Take magnitude: (N, time_frames, freq_bins)
    spectrograms = tf.abs(spectrograms).numpy()
    
    # Transpose to (N, freq_bins, time_frames)
    spectrograms = np.transpose(spectrograms, (0, 2, 1))
    
    # Add channel dimension: (N, freq_bins, time_frames, 1)
    spectrograms = np.expand_dims(spectrograms, axis=-1)
    
    print(f"  ✓ Spectrogram shape: {spectrograms.shape}")
    return spectrograms


def main(use_csi=False, epochs=50, batch_size=512, multi_gpu=False):
    """
    Main CNN detection pipeline.
    
    Args:
        use_csi: Whether to use CSI fusion (default: False)
        epochs: Number of training epochs
        batch_size: Training batch size (512 for H100)
        multi_gpu: Use all available GPUs
    
    Returns:
        success: Whether pipeline succeeded
        results: Dictionary with metrics and metadata
    """
    print("\n" + "#"*70)
    print("🧠 CNN COVERT CHANNEL DETECTION PIPELINE")
    print("#"*70)
    print()
    
    # Setup GPU strategy
    strategy = setup_gpu_strategy(multi_gpu)
    print("#"*70)
    print()
    
    # Initialize directories
    init_directories()
    
    # Results dictionary
    results = {
        'success': False,
        'timestamp': time.time(),
        'config': {
            'detector': 'CNN' + ('+CSI' if use_csi else ''),
            'num_samples': NUM_SAMPLES_PER_CLASS * 2,
            'num_satellites': NUM_SATELLITES_FOR_TDOA,
            'epochs': epochs,
            'batch_size': batch_size,
            'seed': GLOBAL_SEED
        }
    }
    
    # ===== Phase 1: Load Dataset =====
    print(f"\n{'='*70}")
    print("[Phase 1] Loading pre-generated dataset...")
    print(f"{'='*70}")
    
    # 🔧 FIX: Select dataset based on INSIDER_MODE (not auto-detect)
    # Import INSIDER_MODE to select correct dataset
    from config.settings import INSIDER_MODE
    
    # Select scenario-specific dataset
    scenario_name = 'scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b'
    
    # 🔧 FIX: Always prefer latest dataset with scenario name (includes numbered versions)
    import glob
    dataset_files = glob.glob(os.path.join(DATASET_DIR, f"dataset_{scenario_name}*.pkl"))
    if dataset_files:
        # Sort by modification time (newest first) or by name (highest number first)
        dataset_files.sort(key=lambda x: (os.path.getmtime(x), os.path.basename(x)), reverse=True)
        dataset_path = dataset_files[0]  # Latest/newest
        print(f"  → Using latest dataset: {os.path.basename(dataset_path)}")
    else:
        # Fallback to exact name
        dataset_path = f"{DATASET_DIR}/dataset_{scenario_name}.pkl"
        print(f"  → Using: {dataset_path}")
        if not os.path.exists(dataset_path):
            # Final fallback to old naming
            dataset_path = (
                f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
                f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
            )
            print(f"  → Fallback: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        return False, results
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"  ✓ Loaded dataset from {dataset_path}")
    print(f"  → Total samples: {len(dataset['labels'])}")
    print(f"  → Benign: {np.sum(dataset['labels'] == 0)}")
    print(f"  → Attack: {np.sum(dataset['labels'] == 1)}")
    
    # 🔧 TEST 10: Use tx_grids for testing (to check if problem is from channel)
    # Extract data - ✅ REALISTIC: Use rx_grids (post-channel) for training
    # Reason: In practice, detector only sees rx_grid (after channel distortion)
    # Training on rx_grid makes the model realistic and generalizable
    # This is the correct approach for real-world deployment
    USE_PRE_CHANNEL_FOR_TEST = False  # 🔧 TEST 10: Set to False for production (was True for testing)
    
    if USE_PRE_CHANNEL_FOR_TEST and 'tx_grids' in dataset:
        # 🔧 TEST 10: Use pre-channel for testing (to diagnose channel problem)
        X_grids = dataset['tx_grids']  # Pre-channel (for testing only)
        print(f"  🔧 TEST 10: Using PRE-CHANNEL tx_grids (for testing channel impact)")
        print(f"  → This will show if the problem is from channel or CNN")
    elif 'rx_grids' in dataset:
        # Use post-channel for realistic training (both CNN-only and CNN+CSI)
        X_grids = dataset['rx_grids']  # Post-channel (realistic, what detector sees in practice)
        print(f"  ✓ Using POST-CHANNEL rx_grids (realistic training for {'CNN+CSI' if use_csi else 'CNN-only'})")
        print(f"  → Note: This matches real-world scenario where detector only sees post-channel signals")
    elif 'tx_grids' in dataset:
        # Fallback to tx_grids if rx_grids not available (should not happen in normal flow)
        X_grids = dataset['tx_grids']  # Pre-channel (fallback only)
        print(f"  ⚠️ Using PRE-CHANNEL tx_grids (fallback, not realistic for deployment)")
    else:
        X_grids = dataset.get('rx_grids', dataset.get('tx_grids'))
        print(f"  ⚠️ Using available grids")
    
    Y = dataset['labels']
    
    # Apply spectrogram transform if enabled
    if USE_SPECTROGRAM:
        print(f"\n{'='*70}")
        print("[Phase 1.5] Computing Spectrograms...")
        print(f"{'='*70}")
        X_grids = compute_spectrogram(X_grids)
    
    X_csi = None
    csi_fusion_weights = None  # Quality-aware fusion weights
    
    if use_csi and 'csi_est' in dataset and dataset['csi_est'] is not None:
        X_csi = dataset['csi_est']
        print(f"  ✓ CSI_est available: shape {X_csi.shape}")
        # 🔧 DEBUG: Fix CSI shape if needed (remove extra dimension)
        if X_csi.ndim == 4 and X_csi.shape[1] == 1:
            X_csi = np.squeeze(X_csi, axis=1)
            print(f"  🔧 Fixed CSI shape: {X_csi.shape}")
        
        # ===== Quality-Aware Gating for CSI Fusion =====
        # Extract fusion weights from metadata (if available)
        if 'meta' in dataset and isinstance(dataset['meta'], list):
            from config.settings import CSI_CFG
            quality_cfg = CSI_CFG['quality_gating']
            
            csi_fusion_weights = []
            csi_quality_stats = {'good': 0, 'ok': 0, 'bad': 0}
            
            for i, meta_dict in enumerate(dataset['meta']):
                if isinstance(meta_dict, dict):
                    # Get fusion weight from metadata (computed during dataset generation)
                    fusion_weight = meta_dict.get('csi_fusion_weight', 1.0)
                    quality_label = meta_dict.get('csi_quality_label', 'unknown')
                    
                    # If not in metadata, compute from CSI metrics
                    if fusion_weight == 1.0 and 'csi_nmse_db' in meta_dict:
                        from core.csi_estimation import compute_csi_quality_gate
                        csi_info = {
                            'nmse_db': meta_dict.get('csi_nmse_db'),
                            'pilot_mse': meta_dict.get('csi_pilot_mse'),
                            'noise_variance': meta_dict.get('csi_noise_variance'),
                            'P_H': meta_dict.get('csi_P_H'),
                            'phase_comp_deg': meta_dict.get('csi_phase_comp_deg', 0.0),
                            'delay_comp_samples': meta_dict.get('csi_delay_comp_samples', 0)
                        }
                        fusion_weight, quality_label = compute_csi_quality_gate(csi_info, CSI_CFG)
                    
                    csi_fusion_weights.append(fusion_weight)
                    
                    # Statistics
                    if quality_label == 'good':
                        csi_quality_stats['good'] += 1
                    elif quality_label == 'ok':
                        csi_quality_stats['ok'] += 1
                    else:
                        csi_quality_stats['bad'] += 1
                else:
                    csi_fusion_weights.append(1.0)  # Default weight
            
            csi_fusion_weights = np.array(csi_fusion_weights, dtype=np.float32)
            
            # Log quality statistics
            total_samples = len(csi_fusion_weights)
            print(f"  📊 CSI Quality Statistics:")
            print(f"     Good (weight=1.0): {csi_quality_stats['good']}/{total_samples} ({100*csi_quality_stats['good']/total_samples:.1f}%)")
            print(f"     OK (weight=0.6): {csi_quality_stats['ok']}/{total_samples} ({100*csi_quality_stats['ok']/total_samples:.1f}%)")
            print(f"     Bad (weight=0.0): {csi_quality_stats['bad']}/{total_samples} ({100*csi_quality_stats['bad']/total_samples:.1f}%)")
            print(f"     Mean fusion weight: {np.mean(csi_fusion_weights):.3f}")
            
            # Apply quality gating: scale CSI by fusion weights
            if X_csi.ndim == 3:  # (N, symbols, subcarriers)
                # Broadcast weights to match CSI shape
                weights_expanded = csi_fusion_weights[:, np.newaxis, np.newaxis]
                X_csi = X_csi * weights_expanded
                print(f"  ✅ Applied quality-aware gating to CSI (weighted by fusion weights)")
            elif X_csi.ndim == 2:  # (N, features)
                weights_expanded = csi_fusion_weights[:, np.newaxis]
                X_csi = X_csi * weights_expanded
                print(f"  ✅ Applied quality-aware gating to CSI (weighted by fusion weights)")
            
            # Check if any samples have zero weight (CSI disabled)
            disabled_count = np.sum(csi_fusion_weights < 0.01)
            if disabled_count > 0:
                print(f"  ⚠️  {disabled_count} samples have CSI disabled (weight < 0.01) due to poor quality")
        
    elif use_csi and 'csi' in dataset:
        X_csi = dataset['csi']
        print(f"  ✓ CSI (legacy) available: shape {X_csi.shape}")
    elif use_csi:
        print(f"  ⚠️ CSI fusion requested but 'csi' not in dataset!")
        print(f"     Falling back to CNN-only mode.")
        use_csi = False
    
    # 🔍 DEBUG: Check pattern visibility before preprocessing
    print(f"\n  🔍 DEBUG: Pattern visibility check (before preprocessing)...")
    benign_mask = (Y == 0)
    attack_mask = (Y == 1)
    if np.sum(benign_mask) > 0 and np.sum(attack_mask) > 0:
        # Check in tx_grids (pre-channel, where injection happens)
        if 'tx_grids' in dataset:
            tx_grids = dataset['tx_grids']
            benign_tx = np.squeeze(tx_grids[benign_mask][0])
            attack_tx = np.squeeze(tx_grids[attack_mask][0])
            diff_tx = np.abs(attack_tx - benign_tx)
            max_diff_tx = np.max(diff_tx)
            mean_diff_tx = np.mean(diff_tx)
            print(f"    [TX grids (pre-channel)]")
            print(f"      Benign shape: {benign_tx.shape}, mean={np.mean(np.abs(benign_tx)):.6f}")
            print(f"      Attack shape: {attack_tx.shape}, mean={np.mean(np.abs(attack_tx)):.6f}")
            print(f"      Max difference: {max_diff_tx:.6f}")
            print(f"      Mean difference: {mean_diff_tx:.6f}")
            # Check spectral pattern (subcarrier-wise)
            benign_spec = np.mean(np.abs(benign_tx), axis=0)  # Average over symbols
            attack_spec = np.mean(np.abs(attack_tx), axis=0)
            spec_diff = np.abs(attack_spec - benign_spec)
            max_spec_diff = np.max(spec_diff)
            mean_spec_diff = np.mean(spec_diff)
            # Find which subcarriers have the largest difference (should be 24-39 if fixed pattern)
            top_diff_indices = np.argsort(spec_diff)[-10:][::-1]  # Top 10 subcarriers
            print(f"      Spectral diff (subcarrier-wise): max={max_spec_diff:.6f}, mean={mean_spec_diff:.6f}")
            print(f"      Top 10 subcarriers with diff: {top_diff_indices.tolist()}")
            print(f"      Diff values: {spec_diff[top_diff_indices].tolist()}")
            # 🔧 FIX: Check if pattern is in expected subcarriers (24-39, middle band)
            expected_subs = np.arange(24, 40)  # Middle band subcarriers
            # Normalize by number of subcarriers for fair comparison
            num_expected = len(expected_subs)
            num_outside = len(spec_diff) - num_expected
            diff_in_expected = np.sum(spec_diff[expected_subs]) / num_expected  # Average per subcarrier
            outside_indices = np.setdiff1d(np.arange(len(spec_diff)), expected_subs)
            diff_outside_expected = np.sum(spec_diff[outside_indices]) / num_outside  # Average per subcarrier
            print(f"      Diff in subcarriers 24-39 (avg): {diff_in_expected:.6f}")
            print(f"      Diff in subcarriers outside 24-39 (avg): {diff_outside_expected:.6f}")
            if diff_in_expected < diff_outside_expected * 0.8:  # More lenient threshold (0.8 instead of 0.5)
                print(f"      ⚠️  WARNING: Pattern NOT strongly concentrated in 24-39!")
                print(f"         (But this is OK if AUC is high - random variations can be larger)")
            else:
                print(f"      ✓ Pattern is concentrated in subcarriers 24-39")
            if mean_diff_tx < 0.01:
                print(f"      ⚠️  WARNING: Very small difference in tx_grids!")
            else:
                print(f"      ✓ Pattern visible in tx_grids")
        
        # Check in X_grids (what CNN sees)
        benign_sample = X_grids[benign_mask][0]
        attack_sample = X_grids[attack_mask][0]
        benign_sample = np.squeeze(benign_sample)
        attack_sample = np.squeeze(attack_sample)
        diff = np.abs(attack_sample - benign_sample)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"    [X_grids (CNN input)]")
        print(f"      Benign shape: {benign_sample.shape}, mean={np.mean(np.abs(benign_sample)):.6f}")
        print(f"      Attack shape: {attack_sample.shape}, mean={np.mean(np.abs(attack_sample)):.6f}")
        print(f"      Max difference: {max_diff:.6f}")
        print(f"      Mean difference: {mean_diff:.6f}")
        if mean_diff < 0.01:
            print(f"      ⚠️  WARNING: Very small difference! Pattern may be lost in channel.")
        else:
            print(f"      ✓ Pattern difference is visible")
    
    # ===== Phase 2: Physical Metrics Analysis =====
    print(f"\n{'='*70}")
    print("[Phase 2] Physical Metrics Analysis...")
    print(f"{'='*70}")
    
    # Extract scenario info from meta
    # Import INSIDER_MODE after potential update from --scenario argument
    from config.settings import INSIDER_MODE, POWER_PRESERVING_COVERT, COVERT_AMP
    scenario_info = {
        'insider_mode': INSIDER_MODE,
        'injection_point': 'pre-channel',  # Always pre-channel now
        'power_preserving': POWER_PRESERVING_COVERT,
        'covert_amp': COVERT_AMP
    }
    
    # Doppler analysis
    dopplers = []
    if 'meta' in dataset and isinstance(dataset['meta'], list):
        for m in dataset['meta']:
            if isinstance(m, dict) and 'doppler_hz' in m:
                dopplers.append(m['doppler_hz'])
    
    if len(dopplers) > 0:
        dopplers = np.array(dopplers)
        doppler_mean = float(np.mean(dopplers))
        doppler_std = float(np.std(dopplers))
        doppler_min = float(np.min(dopplers))
        doppler_max = float(np.max(dopplers))
        print(f"  ✓ Doppler: mean={doppler_mean:.2f} Hz, std={doppler_std:.2f} Hz")
        print(f"  ✓ Doppler range: [{doppler_min:.2f}, {doppler_max:.2f}] Hz")
        results['doppler_stats'] = {
            'mean_hz': doppler_mean,
            'std_hz': doppler_std,
            'min_hz': doppler_min,
            'max_hz': doppler_max
        }
    else:
        print(f"  ⚠️  No Doppler data in meta")
        results['doppler_stats'] = None
    
    # Power analysis - Use tx_grids for accurate power comparison (pre-channel)
    # X_grids might be rx_grids (post-channel) which has different power
    if 'tx_grids' in dataset:
        tx_grids = dataset['tx_grids']
        benign_tx = np.squeeze(tx_grids[benign_mask])
        attack_tx = np.squeeze(tx_grids[attack_mask])
        benign_power_tx = np.mean(np.abs(benign_tx) ** 2)
        attack_power_tx = np.mean(np.abs(attack_tx) ** 2)
        power_diff_pct = abs(attack_power_tx - benign_power_tx) / benign_power_tx * 100
        print(f"  ✓ Power (tx_grids, pre-channel): benign={benign_power_tx:.6e}, attack={attack_power_tx:.6e}, diff={power_diff_pct:.2f}%")
    else:
        # Fallback to X_grids
        benign_grids = np.squeeze(X_grids[benign_mask])
        attack_grids = np.squeeze(X_grids[attack_mask])
        benign_power = np.mean(np.abs(benign_grids) ** 2)
        attack_power = np.mean(np.abs(attack_grids) ** 2)
        power_diff_pct = abs(attack_power - benign_power) / benign_power * 100
    
    # Also compute from X_grids for comparison
    benign_grids = np.squeeze(X_grids[benign_mask])
    attack_grids = np.squeeze(X_grids[attack_mask])
    benign_power = np.mean(np.abs(benign_grids) ** 2)
    attack_power = np.mean(np.abs(attack_grids) ** 2)
    
    # Per-sample power differences
    benign_powers_per_sample = [np.mean(np.abs(g)**2) for g in benign_grids]
    attack_powers_per_sample = [np.mean(np.abs(g)**2) for g in attack_grids]
    power_diffs_per_sample = []
    for i, ap in enumerate(attack_powers_per_sample):
        if i < len(benign_powers_per_sample):
            diff = abs(ap - benign_powers_per_sample[i]) / (benign_powers_per_sample[i] + 1e-12) * 100
            power_diffs_per_sample.append(diff)
    
    power_diff_mean = np.mean(power_diffs_per_sample) if power_diffs_per_sample else power_diff_pct
    power_diff_std = np.std(power_diffs_per_sample) if power_diffs_per_sample else 0.0
    
    print(f"  ✓ Benign power: {benign_power:.6e} ± {np.std(benign_powers_per_sample):.6e}")
    print(f"  ✓ Attack power: {attack_power:.6e} ± {np.std(attack_powers_per_sample):.6e}")
    print(f"  ✓ Power diff:   {power_diff_pct:.2f}% (mean±std: {power_diff_mean:.2f}±{power_diff_std:.2f}%)")
    
    if power_diff_pct < 5.0:
        print(f"  ✅ Ultra-covert: Power difference < 5% (truly stealthy!)")
    elif power_diff_pct < 10.0:
        print(f"  ✅ Covert: Power difference < 10%")
    else:
        print(f"  ⚠️ Warning: Power difference > 10% (may be detectable)")
    
    results['power_analysis'] = {
        'benign_power': float(benign_power),
        'attack_power': float(attack_power),
        'difference_pct': float(power_diff_pct),
        'difference_mean_pct': float(power_diff_mean),
        'difference_std_pct': float(power_diff_std)
    }
    
    # CSI quality analysis
    if use_csi and X_csi is not None:
        try:
            csi_mag = np.abs(X_csi)
            csi_variance = float(np.var(csi_mag))
            csi_mean = float(np.mean(csi_mag))
            csi_std = float(np.std(csi_mag))
            print(f"  ✓ CSI quality: mean={csi_mean:.6f}, std={csi_std:.6f}, variance={csi_variance:.6e}")
            results['csi_quality'] = {
                'mean': csi_mean,
                'std': csi_std,
                'variance': csi_variance
            }
        except Exception as e:
            print(f"  ⚠️  CSI quality analysis failed: {e}")
            results['csi_quality'] = None
    else:
        results['csi_quality'] = None
    
    # Store scenario info
    results['scenario'] = scenario_info
    
    # Store power_diffs_per_sample for CSV export later
    results['_power_diffs_per_sample'] = power_diffs_per_sample
    
    # ===== Phase 2.5: Pattern Injection Verification =====
    print(f"\n{'='*70}")
    print("[Phase 2.5] Pattern Injection Verification...")
    print(f"{'='*70}")
    
    # Calculate mean absolute difference between attack and benign
    mean_abs_diff = np.mean(np.abs(attack_grids - benign_grids))
    max_abs_diff = np.max(np.abs(attack_grids - benign_grids))
    
    print(f"  ✓ Mean abs difference: {mean_abs_diff:.6f}")
    print(f"  ✓ Max abs difference:  {max_abs_diff:.6f}")
    print(f"  ✓ Dataset keys: {list(dataset.keys())}")
    
    # Check if pattern is actually injected
    if mean_abs_diff < 0.02:
        print(f"  ⚠️  WARNING: Mean difference < 0.02!")
        print(f"      → Pattern may NOT be injected properly")
        print(f"      → Check RANDOMIZE_* settings (must be False)")
        print(f"      → Check USE_SEMI_FIXED_PATTERN = True")
    else:
        print(f"  ✅ Pattern successfully injected (diff ≥ 0.02)")
    
    # Additional stats
    print(f"\n  📊 Additional Statistics:")
    print(f"     Benign mean: {np.mean(benign_grids):.6f}")
    print(f"     Attack mean: {np.mean(attack_grids):.6f}")
    print(f"     Benign std:  {np.std(benign_grids):.6f}")
    print(f"     Attack std:  {np.std(attack_grids):.6f}")
    
    # ===== Phase 3: Train/Test Split =====
    print(f"\n{'='*70}")
    print("[Phase 3] Splitting dataset...")
    print(f"{'='*70}")
    
    test_size = 0.3
    
    if use_csi:
        X_train, X_test, X_csi_train, X_csi_test, y_train, y_test = train_test_split(
            X_grids, X_csi, Y,
            test_size=test_size,
            stratify=Y,
            random_state=GLOBAL_SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_grids, Y,
            test_size=test_size,
            stratify=Y,
            random_state=GLOBAL_SEED
        )
        X_csi_train, X_csi_test = None, None
    
    print(f"  ✓ Training set: {len(y_train)} samples")
    print(f"  ✓ Test set:     {len(y_test)} samples")
    print(f"  ✓ Train benign: {np.sum(y_train == 0)}")
    print(f"  ✓ Train attack: {np.sum(y_train == 1)}")
    print(f"  ✓ Test benign:  {np.sum(y_test == 0)}")
    print(f"  ✓ Test attack:  {np.sum(y_test == 1)}")
    
    # ===== Phase 4: Train CNN Detector =====
    print(f"\n{'='*70}")
    print("[Phase 4] Training CNN detector...")
    print(f"{'='*70}")
    
    detector = CNNDetector(
        use_csi=use_csi,
        learning_rate=0.001,
        dropout_rate=0.3,
        random_state=GLOBAL_SEED,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_LOSS_GAMMA,
        focal_alpha=FOCAL_LOSS_ALPHA
    )
    
    # Train with validation split
    val_size = 0.2
    if use_csi:
        X_tr, X_val, X_csi_tr, X_csi_val, y_tr, y_val = train_test_split(
            X_train, X_csi_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=GLOBAL_SEED
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=GLOBAL_SEED
        )
        X_csi_tr, X_csi_val = None, None
    
    # Debug: Show training data info
    print(f"\n🔍 Training Data Info:")
    print(f"   X_tr shape: {X_tr.shape}")
    print(f"   Attack power: {np.mean(np.abs(X_tr[y_tr==1])):.6f}")
    print(f"   Benign power: {np.mean(np.abs(X_tr[y_tr==0])):.6f}")
    
    history = detector.train(
        X_tr, y_tr, X_csi_train=X_csi_tr,
        X_val=X_val, y_val=y_val, X_csi_val=X_csi_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # ===== Phase 5: Evaluate on Test Set =====
    print(f"\n{'='*70}")
    print("[Phase 5] Evaluating on test set...")
    print(f"{'='*70}")
    
    # 🔧 IMPROVED: Use threshold optimization on validation set
    metrics = detector.evaluate(
        X_test, y_test, X_csi_test=X_csi_test,
        X_val=X_val, y_val=y_val, X_csi_val=X_csi_val
    )

    # Additional physical reporting (already in results from Phase 2)
    if results.get('doppler_stats'):
        metrics['doppler_mean_hz'] = results['doppler_stats']['mean_hz']
        metrics['doppler_std_hz'] = results['doppler_stats']['std_hz']
    if results.get('csi_quality'):
        metrics['csi_variance'] = results['csi_quality']['variance']
    
    print(f"\n📊 Test Set Performance:")
    print(f"  {'='*50}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  {'='*50}")
    
    if metrics['auc'] >= 0.95:
        print(f"  ✅ Excellent detection (AUC ≥ 0.95)")
    elif metrics['auc'] >= 0.85:
        print(f"  ✅ Good detection (AUC ≥ 0.85)")
    elif metrics['auc'] >= 0.70:
        print(f"  ⚠️ Moderate detection (AUC ≥ 0.70)")
    else:
        print(f"  ❌ Poor detection (AUC < 0.70)")
    
    results['metrics'] = metrics
    results['success'] = True
    
    # ===== Phase 6: Save Model =====
    print(f"\n{'='*70}")
    print("[Phase 6] Saving model...")
    print(f"{'='*70}")
    
    # 🔧 FIX: Organize models by scenario
    scenario_folder = 'scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b'
    scenario_model_dir = f"{MODEL_DIR}/{scenario_folder}"
    os.makedirs(scenario_model_dir, exist_ok=True)
    
    model_filename = f"cnn_detector{'_csi' if use_csi else ''}.keras"
    model_path = f"{scenario_model_dir}/{model_filename}"
    detector.save(model_path)
    
    results['model_path'] = model_path
    
    # ===== Save Results =====
    # 🔧 FIX: Organize results by scenario (scenario_a for 'sat', scenario_b for 'ground')
    scenario_folder = 'scenario_a' if INSIDER_MODE == 'sat' else 'scenario_b'
    scenario_result_dir = f"{RESULT_DIR}/{scenario_folder}"
    os.makedirs(scenario_result_dir, exist_ok=True)
    
    detector_suffix = '_csi' if use_csi else ''
    result_filename = f"detection_results_cnn{detector_suffix}.json"
    result_path = f"{scenario_result_dir}/{result_filename}"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {result_path}")
    
    # ===== Export CSV Meta Log =====
    if 'meta' in dataset and isinstance(dataset['meta'], list):
        try:
            import pandas as pd
            csv_data = []
            for i, m in enumerate(dataset['meta']):
                if isinstance(m, dict):
                    row = {
                        'sample_idx': i,
                        'label': int(Y[i]) if i < len(Y) else -1,
                        'doppler_hz': m.get('doppler_hz', 0.0),
                        'insider_mode': m.get('insider_mode', INSIDER_MODE),
                        'power_preserving': m.get('power_preserving', POWER_PRESERVING_COVERT)
                    }
                    # Add per-sample power diff if available
                    power_diffs = results.get('_power_diffs_per_sample', [])
                    if i < len(power_diffs):
                        row['power_diff_pct'] = power_diffs[i]
                    csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_filename = f"run_meta_log{detector_suffix}.csv"
                csv_path = f"{scenario_result_dir}/{csv_filename}"
                df.to_csv(csv_path, index=False)
                print(f"✓ Meta log CSV saved to {csv_path}")
        except Exception as e:
            print(f"  ⚠️  CSV export failed: {e}")
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    print("📋 PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Dataset:        {NUM_SAMPLES_PER_CLASS * 2} samples")
    print(f"  Scenario:       {INSIDER_MODE} (insider@{'satellite' if INSIDER_MODE=='sat' else 'ground'})")
    print(f"  Injection:      pre-channel")
    print(f"  Power-preserving: {'on' if POWER_PRESERVING_COVERT else 'off'}")
    print(f"  Detector:       CNN{'+CSI' if use_csi else ''}")
    # Report stable, interpretable power numbers
    print(f"  Power diff:     {power_diff_pct:.2f}%")
    print(f"  Power (benign): {benign_power:.6e} (std {np.std(benign_powers_per_sample):.6e})")
    print(f"  Power (attack): {attack_power:.6e} (std {np.std(attack_powers_per_sample):.6e})")
    if results.get('doppler_stats'):
        dd = results['doppler_stats']
        print(f"  Doppler:        {dd['mean_hz']:.2f} Hz (std {dd['std_hz']:.2f} Hz)")
    if results.get('csi_quality'):
        cq = results['csi_quality']
        print(f"  CSI variance:   {cq['variance']:.6e}")
    print(f"  Test AUC:       {metrics['auc']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1']:.4f}")
    print(f"  Model saved:    {model_path}")
    print(f"  Results saved:  {result_path}")
    print(f"{'='*70}")
    
    return True, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN-based covert channel detection')
    parser.add_argument('--use-csi', action='store_true',
                       help='Enable CSI fusion (multi-modal)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Training batch size (default: 512 for H100)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs (MirroredStrategy)')
    parser.add_argument('--scenario', type=str, choices=['sat', 'ground', 'a', 'b'],
                       help='Scenario: "sat"/"a" (Scenario A) or "ground"/"b" (Scenario B). If not provided, uses INSIDER_MODE from settings.py')
    
    args = parser.parse_args()
    
    # Update INSIDER_MODE if scenario is provided
    if args.scenario:
        scenario_mode = 'sat' if args.scenario in ['sat', 'a'] else 'ground'
        from run_all_scenarios import update_settings_file
        update_settings_file('INSIDER_MODE', f"'{scenario_mode}'")
        # Reload settings to get updated INSIDER_MODE
        import importlib
        from config import settings
        importlib.reload(settings)
        from config.settings import INSIDER_MODE
        print(f"  ✓ Scenario set to: {scenario_mode} (INSIDER_MODE={INSIDER_MODE})")
    
    try:
        success, results = main(
            use_csi=args.use_csi,
            epochs=args.epochs,
            batch_size=args.batch_size,
            multi_gpu=args.multi_gpu
        )
        
        if success:
            print("\n✅ CNN detection pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ CNN detection pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
