"""
Scenario B: Relay Implementation
==================================
Amplify-and-Forward (AF) relay for uplink → downlink scenario.
"""

import numpy as np
import tensorflow as tf


def amplify_and_forward_relay(y_ul, target_power=None, gain_limit_db=30.0):
    """
    Amplify-and-Forward relay with AGC (Automatic Gain Control).
    
    Args:
        y_ul: Received signal at relay (uplink)
        target_power: Target output power (if None, preserves input power)
        gain_limit_db: Maximum gain in dB (prevents excessive amplification)
    
    Returns:
        y_dl: Amplified signal for downlink transmission
        gain_applied: Applied gain (for logging)
    """
    # Compute input power
    input_power = tf.reduce_mean(tf.abs(y_ul)**2)
    input_power = tf.maximum(input_power, 1e-20)  # Prevent division by zero
    
    # Determine target power
    if target_power is None:
        target_power = input_power  # Preserve power by default
    
    # Compute required gain
    gain_linear = target_power / input_power
    gain_db = 10.0 * tf.math.log(gain_linear + 1e-12) / tf.math.log(10.0)
    
    # Limit gain to prevent excessive amplification
    gain_limit_linear = 10.0**(gain_limit_db / 10.0)
    gain_linear = tf.minimum(gain_linear, gain_limit_linear)
    
    # Apply gain
    y_dl = y_ul * tf.cast(tf.sqrt(gain_linear), dtype=y_ul.dtype)
    
    return y_dl, float(gain_linear.numpy())


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

