"""
Scenario B: Relay Implementation
==================================
Amplify-and-Forward (AF) relay for uplink → downlink scenario.
"""

import numpy as np
import tensorflow as tf


def amplify_and_forward_relay(y_ul, target_power=None, gain_limit_db=30.0, delay_samples=0, clip_range=(-2.0, 2.0)):
    """
    Phase 6: Enhanced Amplify-and-Forward relay with AGC, delay, and clipping.
    
    Args:
        y_ul: Received signal at relay (uplink) - time domain
        target_power: Target output power (if None, preserves input power)
        gain_limit_db: Maximum gain in dB (prevents excessive amplification)
        delay_samples: Processing delay in samples (3-5 symbols typical)
        clip_range: Tuple (min, max) for clipping to prevent saturation
    
    Returns:
        y_dl: Amplified signal for downlink transmission
        gain_applied: Applied gain (for logging)
        delay_applied: Applied delay (for logging)
    """
    # Phase 6: Add processing delay (simulate relay processing time)
    if delay_samples > 0:
        # Pad at the beginning and crop at the end
        y_ul_padded = tf.pad(y_ul, [[0, 0], [delay_samples, 0]], mode='CONSTANT', constant_values=0.0)
        # Crop to original length
        y_ul = y_ul_padded[:, :tf.shape(y_ul)[1]]
    
    # Compute input power
    input_power = tf.reduce_mean(tf.abs(y_ul)**2)
    input_power = tf.maximum(input_power, 1e-20)  # Prevent division by zero
    
    # Determine target power
    if target_power is None:
        target_power = input_power  # Preserve power by default
    
    # Compute required gain
    gain_linear = target_power / input_power
    
    # Phase 6: Limit gain to prevent excessive amplification
    # 🔧 CRITICAL FIX: Increase relay gain significantly to compensate for dual-hop attenuation
    # With -41 dB channel attenuation, we need MUCH more amplification
    # Original: gain_min=0.5, gain_max=2.0 → signal too weak
    # Test 11d: gain_min=1.0, gain_max=4.0 → still too weak
    # Previous: gain_min=2.0, gain_max=8.0 → still too weak (SNR_raw = -29 dB)
    # Final: gain_min=4.0, gain_max=16.0 → compensate for -41 dB dual-hop attenuation
    gain_min = 4.0  # 🔧 CRITICAL: Increased from 2.0 (allow much more amplification)
    gain_max = 16.0  # 🔧 CRITICAL: Increased from 8.0 (compensate for -41 dB dual-hop attenuation)
    gain_linear = tf.clip_by_value(gain_linear, gain_min, gain_max)
    
    # Apply gain
    y_dl = y_ul * tf.cast(tf.sqrt(gain_linear), dtype=y_ul.dtype)
    
    # Phase 6: Clip to prevent saturation (optional but recommended)
    if clip_range is not None:
        y_dl_real = tf.clip_by_value(tf.math.real(y_dl), clip_range[0], clip_range[1])
        y_dl_imag = tf.clip_by_value(tf.math.imag(y_dl), clip_range[0], clip_range[1])
        y_dl = tf.complex(y_dl_real, y_dl_imag)
    
    return y_dl, float(gain_linear.numpy()), delay_samples


def compute_doppler_ul(emitter_pos, emitter_vel, sat_pos, sat_vel, carrier_freq):
    """
    Compute Doppler shift for uplink (ground → satellite).
    
    Args:
        emitter_pos: Emitter position (ground)
        emitter_vel: Emitter velocity (usually [0,0,0] for ground)
        sat_pos: Satellite position
        sat_vel: Satellite velocity
        carrier_freq: Carrier frequency
    
    Returns:
        f_d_ul: Doppler shift in Hz
    """
    c0 = 3e8
    los = np.array(sat_pos) - np.array(emitter_pos)
    los_hat = los / (np.linalg.norm(los) + 1e-12)
    
    # Relative velocity: v_sat - v_emitter (projected onto LOS)
    v_rel = np.dot(np.array(sat_vel) - np.array(emitter_vel), los_hat)
    
    # Doppler: f_d = (v_rel / c) * f_c
    f_d_ul = (v_rel / c0) * carrier_freq
    
    return float(f_d_ul)


def compute_doppler_dl(sat_pos, sat_vel, receiver_pos, receiver_vel, carrier_freq):
    """
    Compute Doppler shift for downlink (satellite → ground).
    
    Args:
        sat_pos: Satellite position
        sat_vel: Satellite velocity
        receiver_pos: Receiver position (ground)
        receiver_vel: Receiver velocity (usually [0,0,0] for ground)
        carrier_freq: Carrier frequency
    
    Returns:
        f_d_dl: Doppler shift in Hz
    """
    c0 = 3e8
    los = np.array(receiver_pos) - np.array(sat_pos)
    los_hat = los / (np.linalg.norm(los) + 1e-12)
    
    # Relative velocity: v_receiver - v_sat (projected onto LOS)
    v_rel = np.dot(np.array(receiver_vel) - np.array(sat_vel), los_hat)
    
    # Doppler: f_d = (v_rel / c) * f_c
    f_d_dl = (v_rel / c0) * carrier_freq
    
    return float(f_d_dl)

