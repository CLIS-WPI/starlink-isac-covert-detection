#!/usr/bin/env python3
"""
🔧 Phase 1: Enhanced Dataset Generator with Parameter Diversity
===============================================================
Generates dataset with diverse configurations (SNR, amplitude, Doppler, pattern, subband).
"""

import numpy as np
import tensorflow as tf
from config.settings import (
    INSIDER_MODE, POWER_PRESERVING_COVERT, COVERT_AMP, CARRIER_FREQUENCY
)
from core.covert_injection_phase1 import inject_covert_channel_fixed_phase1

# Try to import NTN utilities for full channel model
try:
    from sionna.phy.channel.tr38811 import utils as tr811_utils
    NTN_AVAILABLE = True
except:
    NTN_AVAILABLE = False

from sionna.phy.channel import (
    time_to_ofdm_channel,
    cir_to_time_channel,
    time_lag_discrete_time_channel
)


def generate_dataset_phase1(isac_system, num_samples, num_satellites,
                           phase1_configs, start_idx=0,
                           tle_path=None, inject_attack_into_pathb=True):
    """
    Phase 1: Generate dataset with diverse configurations.
    
    Args:
        isac_system: ISACSystem instance
        num_samples: Total number of samples to generate
        num_satellites: Number of satellites
        phase1_configs: List of config dicts (snr_db, covert_amp, doppler_scale, pattern, subband_mode)
        start_idx: Starting sample index (for metadata)
        tle_path: Optional TLE file path
        inject_attack_into_pathb: Whether to inject attack
    
    Returns:
        dict: Dataset with tx_grids, rx_grids, labels, csi_est, meta
    """
    from core.dataset_generator import (
        check_angular_diversity
    )
    
    # Initialize containers
    all_tx_grids = []
    all_rx_grids = []
    all_csi_est = []
    all_labels = []
    all_meta = []
    
    num_benign = num_samples // 2
    num_attack = num_samples - num_benign
    
    # Distribute configs across samples
    config_idx = 0
    num_configs = len(phase1_configs)
    
    print(f"[Phase1] Generating {num_samples} samples with {num_configs} configurations...")
    
    for sample_idx in range(num_samples):
        is_attack = (sample_idx >= num_benign)
        
        # Select config for this sample (round-robin)
        config = phase1_configs[config_idx % num_configs]
        config_idx += 1
        
        # Extract Phase 1 parameters
        snr_db = config['snr_db']
        covert_amp = config['covert_amp']
        doppler_scale = config['doppler_scale']
        pattern = config['pattern']
        subband_mode = config['subband_mode']
        
        # Generate topology (inline, similar to dataset_generator.py)
        base_positions = []
        base_velocities = []
        grid_spacing = 100e3
        
        def get_random_velocity():
            v_mag = 7500.0 + np.random.uniform(-500, 500)  # m/s
            v_vec = np.random.randn(3)
            v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-12) * v_mag
            return v_vec
        
        if num_satellites == 4:
            base_positions_coords = [
                np.array([0.0, 0.0, 600e3]),
                np.array([grid_spacing, 0.0, 600e3]),
                np.array([0.0, grid_spacing, 600e3]),
                np.array([grid_spacing, grid_spacing, 600e3]),
            ]
            for pos in base_positions_coords:
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions.append(np.array([pos[0], pos[1], pos[2] + altitude_offset]))
                base_velocities.append(get_random_velocity())
        elif num_satellites == 12:
            user_center_x, user_center_y = 75e3, 75e3
            shells = [545e3, 575e3, 345e3]
            shell_weights = [0.5, 0.35, 0.15]
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(100e3, 140e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights) + np.random.uniform(-30e3, 30e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity())
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.pi/4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(220e3, 280e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights) + np.random.uniform(-40e3, 40e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity())
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(380e3, 450e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights) + np.random.uniform(-50e3, 50e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity())
        else:
            # Generic grid
            side = int(np.ceil(np.sqrt(num_satellites)))
            for j in range(num_satellites):
                x = (j % side) * grid_spacing
                y_p = (j // side) * grid_spacing
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions.append(np.array([x, y_p, 600e3 + altitude_offset]))
                base_velocities.append(get_random_velocity())
        
        # Generate emitter location (ground-based, random within ±500 km)
        emitter_offset_x = np.random.uniform(-500e3, 500e3)
        emitter_offset_y = np.random.uniform(-500e3, 500e3)
        emitter_pos = np.array([emitter_offset_x, emitter_offset_y, 0.0])  # Ground level
        
        # Select attacked satellite (for attack samples)
        if is_attack:
            attacked_sat_idx = np.random.randint(0, num_satellites)
            attacked_sat_pos = base_positions[attacked_sat_idx]
            attacked_sat_vel = base_velocities[attacked_sat_idx]
        else:
            attacked_sat_idx = None
            attacked_sat_pos = None
            attacked_sat_vel = None
        
        # Generate clean TX signal
        total_info_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
        num_codewords = int(np.ceil(total_info_bits / isac_system.k))
        
        b = isac_system.binary_source([1, total_info_bits])
        c_blocks = [isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) 
                    for j in range(num_codewords)]
        c = tf.concat(c_blocks, axis=1)[:, :total_info_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [1, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_grid_clean = isac_system.rg_mapper(x)
        
        # Phase 1: Inject covert signal with diverse patterns
        power_before = float(np.mean(np.abs(tx_grid_clean.numpy())**2))
        power_after = power_before
        power_diff_pct = 0.0
        injection_info = None
        
        if is_attack:
            try:
                tx_grid_np = tx_grid_clean.numpy()
                tx_grid_injected_np, injection_info = inject_covert_channel_fixed_phase1(
                    tx_grid_np,
                    isac_system.rg,
                    pattern=pattern,
                    subband_mode=subband_mode,
                    covert_amp=covert_amp,
                    power_preserving=POWER_PRESERVING_COVERT
                )
                tx_grid_injected = tf.constant(tx_grid_injected_np, dtype=tf.complex64)
                power_after = float(np.mean(np.abs(tx_grid_injected_np)**2))
                power_diff_pct = abs(power_after - power_before) / (power_before + 1e-12) * 100.0
            except Exception as e:
                print(f"[Phase1][WARN] Injection failed for sample {sample_idx}: {e}")
                tx_grid_injected = tx_grid_clean
                injection_info = {'error': str(e)}
        else:
            tx_grid_injected = tx_grid_clean
        
        # Store TX grid
        g = tx_grid_injected.numpy().astype(np.complex64)
        all_tx_grids.append((start_idx + sample_idx, g))
        
        # Simulate channel (simplified - use first satellite)
        sat_pos = base_positions[0]
        sat_vel = base_velocities[0]
        
        # Initialize Scenario B metadata (will be populated if INSIDER_MODE == 'ground')
        scenario_b_meta = None
        f_d = 0.0  # Default Doppler
        f_d_ul = 0.0
        f_d_dl = 0.0
        gain_relay = 1.0
        delay_applied = 0
        snr_ul = snr_db
        snr_dl = snr_db
        
        # ============================================================
        # SCENARIO B: Dual-hop (Uplink → Relay → Downlink)
        # ============================================================
        if INSIDER_MODE == 'ground':
            from core.scenario_b_relay import (
                amplify_and_forward_relay,
                compute_doppler_ul,
                compute_doppler_dl
            )
            
            # UPLINK: Ground → Satellite
            emitter_pos_ul = emitter_pos  # Ground emitter
            emitter_vel_ul = np.array([0.0, 0.0, 0.0])  # Ground is stationary
            
            # Compute UL Doppler
            try:
                f_d_ul_base = compute_doppler_ul(
                    emitter_pos_ul, emitter_vel_ul,
                    sat_pos, sat_vel,
                    CARRIER_FREQUENCY
                )
                f_d_ul = f_d_ul_base * doppler_scale  # Apply Phase 1 Doppler scale
            except Exception:
                f_d_ul = 0.0
            
            # 🔧 Phase 6: Generate FULL UL channel (independent channel model)
            tx_time = isac_system.modulator(tx_grid_injected)
            tx_time_flat = tf.squeeze(tx_time)
            
            # Generate UL channel with full channel model
            if NTN_AVAILABLE and hasattr(isac_system.CHANNEL_MODEL, 'set_topology'):
                ut_pos_ul = tf.constant([[emitter_pos_ul]], dtype=tf.float32)
                bs_pos_ul = tf.constant([[sat_pos]], dtype=tf.float32)
                bs_vel_ul = tf.constant([[sat_vel]], dtype=tf.float32)
                ut_vel_ul = tf.zeros_like(ut_pos_ul)
                try:
                    isac_system.CHANNEL_MODEL.set_topology(ut_pos_ul, bs_pos_ul, ut_vel_ul, bs_vel_ul)
                    a_ul, tau_ul = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
                except Exception:
                    a_ul, tau_ul = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)
            else:
                a_ul, tau_ul = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)
            
            # Sanitize UL CIR
            a_ul_is_finite = tf.logical_and(
                tf.math.is_finite(tf.math.real(a_ul)),
                tf.math.is_finite(tf.math.imag(a_ul))
            )
            a_ul = tf.where(a_ul_is_finite, a_ul, tf.zeros_like(a_ul))
            tau_ul = tf.where(tf.math.is_finite(tau_ul), tau_ul, tf.zeros_like(tau_ul))
            
            # UL CIR → time channel → OFDM channel
            l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
            h_time_ul = cir_to_time_channel(isac_system.rg.bandwidth, a_ul, tau_ul, l_min, l_max, normalize=True)
            
            if h_time_ul.shape[-2] == 1:
                mult = [1] * len(h_time_ul.shape)
                mult[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time_ul = tf.tile(h_time_ul, mult)
            
            h_freq_ul = time_to_ofdm_channel(h_time_ul, isac_system.rg, l_min)
            h_freq_ul = tf.reduce_mean(h_freq_ul, axis=4)
            h_freq_ul = tf.reduce_mean(h_freq_ul, axis=2)
            h_f_ul = h_freq_ul[:, 0, 0, :, :]
            h_f_ul = tf.expand_dims(h_f_ul, axis=1)
            h_f_ul = tf.expand_dims(h_f_ul, axis=2)
            
            # Apply UL channel in frequency domain
            tx_grid_expanded = tf.expand_dims(tx_grid_injected, axis=0) if len(tx_grid_injected.shape) == 4 else tx_grid_injected
            y_grid_ul = tx_grid_expanded * h_f_ul
            snr_ul = snr_db  # Use same SNR for UL
            
            # Convert to time domain for UL
            y_time_ul = isac_system.modulator(y_grid_ul)
            y_time_ul_flat = tf.squeeze(y_time_ul)
            
            # Apply UL Doppler
            y_time_ul_dopp = isac_system.apply_doppler_time(y_time_ul_flat, float(f_d_ul))
            
            # Add UL noise
            rx_pow_ul = tf.reduce_mean(tf.abs(y_time_ul_dopp)**2)
            rx_pow_ul = tf.maximum(rx_pow_ul, 1e-20)
            esn0_ul = 10.0**(snr_ul/10.0)
            sigma2_ul = rx_pow_ul / (esn0_ul + 1e-12)
            std_ul = tf.sqrt(tf.cast(tf.maximum(sigma2_ul / 2.0, 1e-30), tf.float32))
            n_ul = tf.complex(
                tf.random.normal(tf.shape(y_time_ul_dopp), stddev=std_ul),
                tf.random.normal(tf.shape(y_time_ul_dopp), stddev=std_ul)
            )
            y_time_ul_noisy = y_time_ul_dopp + n_ul
            
            # ============================================================
            # RELAY: Amplify-and-Forward with AGC + Delay
            # ============================================================
            delay_samples = np.random.randint(3, 6)  # 3-5 samples delay
            target_power_relay = tf.reduce_mean(tf.abs(y_time_ul_noisy)**2)
            
            y_time_relay_batch = tf.expand_dims(y_time_ul_noisy, axis=0)
            y_time_relay, gain_relay, delay_applied = amplify_and_forward_relay(
                y_time_relay_batch,
                target_power=target_power_relay,
                gain_limit_db=30.0,
                delay_samples=delay_samples,
                clip_range=(-2.0, 2.0)
            )
            y_time_relay = tf.squeeze(y_time_relay, axis=0)
            
            # ============================================================
            # DOWNLINK: Satellite → Ground Receiver
            # ============================================================
            receiver_pos = np.array([50e3, 50e3, 0.0])  # Ground receiver
            receiver_vel = np.array([0.0, 0.0, 0.0])
            
            # Compute DL Doppler
            try:
                f_d_dl_base = compute_doppler_dl(
                    sat_pos, sat_vel,
                    receiver_pos, receiver_vel,
                    CARRIER_FREQUENCY
                )
                f_d_dl = f_d_dl_base * doppler_scale  # Apply Phase 1 Doppler scale
            except Exception:
                f_d_dl = 0.0
            
            # 🔧 Phase 6: Generate FULL DL channel (independent channel model)
            # Convert relay output to frequency domain for channel application
            y_grid_relay = isac_system.demodulator(tf.expand_dims(y_time_relay, axis=0))
            y_grid_relay_squeezed = tf.squeeze(y_grid_relay, axis=0) if len(y_grid_relay.shape) > 2 else y_grid_relay
            
            # Generate DL channel with full channel model
            if NTN_AVAILABLE and hasattr(isac_system.CHANNEL_MODEL, 'set_topology'):
                ut_pos_dl = tf.constant([[receiver_pos]], dtype=tf.float32)
                bs_pos_dl = tf.constant([[sat_pos]], dtype=tf.float32)
                bs_vel_dl = tf.constant([[sat_vel]], dtype=tf.float32)
                ut_vel_dl = tf.zeros_like(ut_pos_dl)
                try:
                    isac_system.CHANNEL_MODEL.set_topology(ut_pos_dl, bs_pos_dl, ut_vel_dl, bs_vel_dl)
                    a_dl, tau_dl = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
                except Exception:
                    a_dl, tau_dl = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)
            else:
                a_dl, tau_dl = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)
            
            # Sanitize DL CIR
            a_dl_is_finite = tf.logical_and(
                tf.math.is_finite(tf.math.real(a_dl)),
                tf.math.is_finite(tf.math.imag(a_dl))
            )
            a_dl = tf.where(a_dl_is_finite, a_dl, tf.zeros_like(a_dl))
            tau_dl = tf.where(tf.math.is_finite(tau_dl), tau_dl, tf.zeros_like(tau_dl))
            
            # DL CIR → time channel → OFDM channel
            h_time_dl = cir_to_time_channel(isac_system.rg.bandwidth, a_dl, tau_dl, l_min, l_max, normalize=True)
            
            if h_time_dl.shape[-2] == 1:
                mult = [1] * len(h_time_dl.shape)
                mult[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time_dl = tf.tile(h_time_dl, mult)
            
            h_freq_dl = time_to_ofdm_channel(h_time_dl, isac_system.rg, l_min)
            h_freq_dl = tf.reduce_mean(h_freq_dl, axis=4)
            h_freq_dl = tf.reduce_mean(h_freq_dl, axis=2)
            h_f_dl = h_freq_dl[:, 0, 0, :, :]
            h_f_dl = tf.expand_dims(h_f_dl, axis=1)
            h_f_dl = tf.expand_dims(h_f_dl, axis=2)
            
            # Apply DL channel in frequency domain
            y_grid_relay_expanded = tf.expand_dims(y_grid_relay_squeezed, axis=0) if len(y_grid_relay_squeezed.shape) == 2 else y_grid_relay_squeezed
            y_grid_dl = y_grid_relay_expanded * h_f_dl
            snr_dl = snr_db  # Use same SNR for DL
            
            # Convert to time domain for DL
            y_time_dl = isac_system.modulator(y_grid_dl)
            y_time_dl_flat = tf.squeeze(y_time_dl)
            
            # Apply DL Doppler
            y_time_dl_dopp = isac_system.apply_doppler_time(y_time_dl_flat, float(f_d_dl))
            
            # Add DL noise
            rx_pow_dl = tf.reduce_mean(tf.abs(y_time_dl_dopp)**2)
            rx_pow_dl = tf.maximum(rx_pow_dl, 1e-20)
            esn0_dl = 10.0**(snr_dl/10.0)
            sigma2_dl = rx_pow_dl / (esn0_dl + 1e-12)
            std_dl = tf.sqrt(tf.cast(tf.maximum(sigma2_dl / 2.0, 1e-30), tf.float32))
            n_dl = tf.complex(
                tf.random.normal(tf.shape(y_time_dl_dopp), stddev=std_dl),
                tf.random.normal(tf.shape(y_time_dl_dopp), stddev=std_dl)
            )
            y_time_noisy = y_time_dl_dopp + n_dl
            
            # Store Scenario B metadata
            scenario_b_meta = {
                'fd_ul': float(f_d_ul),
                'fd_dl': float(f_d_dl),
                'G_r_mean': float(gain_relay),
                'delay_samples': int(delay_applied),
                'snr_ul': float(snr_ul),
                'snr_dl': float(snr_dl)
            }
            
            # Use DL Doppler for main doppler_hz field
            f_d = f_d_dl
        
        # ============================================================
        # SCENARIO A: Single downlink channel
        # ============================================================
        else:
            # Apply channel (simplified)
            tx_time = isac_system.modulator(tx_grid_injected)
            tx_time_flat = tf.squeeze(tx_time)
            
            # Compute Doppler with scale factor
            c0 = 3e8
            em_pos = attacked_sat_pos if is_attack else emitter_pos
            los = np.array(sat_pos) - np.array(em_pos)
            los_hat = los / (np.linalg.norm(los) + 1e-12)
            v_rel = np.dot(np.array(sat_vel), los_hat)
            f_d_base = (v_rel / c0) * CARRIER_FREQUENCY
            f_d = f_d_base * doppler_scale  # Apply Phase 1 Doppler scale
            
            # Apply Doppler
            y_time_dopp = isac_system.apply_doppler_time(tx_time_flat, float(f_d))
            
            # Add noise with specified SNR
            rx_pow = tf.reduce_mean(tf.abs(y_time_dopp)**2)
            esn0 = 10.0**(snr_db/10.0)
            sigma2 = rx_pow / (esn0 + 1e-12)
            std_td = tf.sqrt(tf.cast(tf.maximum(sigma2 / 2.0, 1e-30), tf.float32))
            n_td = tf.complex(
                tf.random.normal(tf.shape(y_time_dopp), stddev=std_td),
                tf.random.normal(tf.shape(y_time_dopp), stddev=std_td)
            )
            y_time_noisy = y_time_dopp + n_td
        # Expand dims for demodulator: (N,) -> (1, N) for batch dimension
        y_time_noisy_batch = tf.expand_dims(y_time_noisy, axis=0)
        y_grid_noisy = isac_system.demodulator(y_time_noisy_batch)
        
        # Store RX grid - squeeze to remove batch dimension if present
        # demodulator returns shape like [1, symbols, subcarriers] or [1, 1, 1, symbols, subcarriers]
        rx_grid_shape = y_grid_noisy.shape
        if len(rx_grid_shape) == 5:
            # [1, 1, 1, symbols, subcarriers] -> [symbols, subcarriers]
            rx_grid_squeezed = tf.squeeze(y_grid_noisy, axis=[0, 1, 2])
        elif len(rx_grid_shape) == 3:
            # [1, symbols, subcarriers] -> [symbols, subcarriers]
            rx_grid_squeezed = tf.squeeze(y_grid_noisy, axis=0)
        else:
            # Already 2D [symbols, subcarriers]
            rx_grid_squeezed = y_grid_noisy
        
        rx_grid_np = rx_grid_squeezed.numpy().astype(np.complex64)
        
        # Fix: Handle shape correctly - tx_grid_injected is [1, 1, 1, symbols, subcarriers]
        tx_grid_shape = tx_grid_injected.shape
        if len(tx_grid_shape) == 5:
            tx_squeezed = tf.squeeze(tx_grid_injected, axis=[0, 1, 2])
        elif len(tx_grid_shape) == 3:
            tx_squeezed = tf.squeeze(tx_grid_injected, axis=0)
        else:
            tx_squeezed = tx_grid_injected
        
        tx_grid_np = tx_squeezed.numpy().astype(np.complex64)
        rx_squeezed = rx_grid_squeezed  # Already squeezed above
        
        pilot_symbols = [2, 7]
        num_symbols, num_subcarriers = int(rx_squeezed.shape[0]), int(rx_squeezed.shape[1])
        
        # ============================================================
        # ✅ NEW: MMSE Equalization for Scenario B (Pilot-assisted)
        # ============================================================
        if INSIDER_MODE == 'ground':
            # Use LMMSE estimation for better channel estimate
            from core.csi_estimation import (
                estimate_csi_lmmse_simple,
                mmse_equalize,
                compute_pattern_preservation
            )
            
            # Estimate CSI using LMMSE (better than LS for low SNR)
            h_est_np = estimate_csi_lmmse_simple(
                tx_grid_np, rx_grid_np,
                pilot_symbols=pilot_symbols,
                snr_db=snr_dl  # Use downlink SNR for Scenario B
            )
            
            # Apply MMSE equalization
            rx_grid_eq, eq_info = mmse_equalize(
                rx_grid_np, h_est_np,
                snr_db=snr_dl,  # Use downlink SNR
                alpha_reg=None,  # Auto-compute based on SNR
                blend_factor=None  # Auto-compute based on SNR
            )
            
            # Monitor pattern preservation (for first few samples)
            if sample_idx < 10:  # Only for first 10 samples to avoid overhead
                try:
                    preservation_metrics = compute_pattern_preservation(
                        tx_grid_np, rx_grid_np, rx_grid_eq,
                        target_subcarriers=np.arange(24, 40)
                    )
                    # Log if preservation is poor
                    if preservation_metrics['preservation_eq'] < 0.6:
                        print(f"  ⚠️  Sample {sample_idx}: Poor pattern preservation "
                              f"(preservation={preservation_metrics['preservation_eq']:.3f})")
                except Exception as e:
                    pass  # Skip if computation fails
            
            # Use equalized signal for Scenario B
            rx_grid_np = rx_grid_eq.astype(np.complex64)
            
            # Store equalization info in metadata
            if scenario_b_meta is None:
                scenario_b_meta = {}
            scenario_b_meta['eq_alpha'] = eq_info['alpha_used']
            scenario_b_meta['eq_snr_db'] = eq_info['snr_eq_db']
            scenario_b_meta['eq_snr_raw_db'] = eq_info.get('snr_raw_db', snr_dl)  # 🔧 FINAL CHECK: Pre-EQ SNR
            scenario_b_meta['eq_snr_improvement_db'] = eq_info.get('snr_improvement_db', 0.0)  # 🔧 FINAL CHECK: SNR gain
            scenario_b_meta['eq_blend_factor'] = eq_info['blend_factor']
        else:
            # Scenario A: Use simple LS estimation (no equalization needed)
            h_est_np = np.zeros((num_symbols, num_subcarriers), dtype=np.complex64)
            for sym_idx in pilot_symbols:
                if sym_idx < num_symbols:
                    for sc_idx in range(num_subcarriers):
                        tx_pilot = tx_squeezed[sym_idx, sc_idx]
                        rx_pilot = rx_squeezed[sym_idx, sc_idx]
                        if tf.abs(tx_pilot) > 1e-9:
                            h_est_np[sym_idx, sc_idx] = (rx_pilot / tx_pilot).numpy()
        
        all_rx_grids.append((start_idx + sample_idx, rx_grid_np))
        all_csi_est.append((start_idx + sample_idx, h_est_np))
        
        # Store label
        all_labels.append((start_idx + sample_idx, 1 if is_attack else 0))
        
        # Store metadata
        meta = {
            'sample_idx': start_idx + sample_idx,
            'label': 1 if is_attack else 0,
            'insider_mode': INSIDER_MODE,
            'snr_db': snr_db,
            'covert_amp': covert_amp,
            'doppler_scale': doppler_scale,
            'doppler_hz': float(f_d),
            'pattern': pattern,
            'subband_mode': subband_mode,
            'power_before': power_before,
            'power_after': power_after,
            'power_diff_pct': power_diff_pct,
            'power_preserving': POWER_PRESERVING_COVERT,
            'injection_info': injection_info if injection_info else {}
        }
        
        # Phase 6: Add Scenario B metadata if available
        if scenario_b_meta:
            meta.update(scenario_b_meta)
        
        all_meta.append((start_idx + sample_idx, meta))
        
        if (sample_idx + 1) % 100 == 0:
            print(f"  Generated {sample_idx + 1}/{num_samples} samples...")
    
    # Sort by index and extract data
    all_tx_grids = [x[1] for x in sorted(all_tx_grids, key=lambda z: z[0])]
    all_rx_grids = [x[1] for x in sorted(all_rx_grids, key=lambda z: z[0])]
    all_csi_est = [x[1] for x in sorted(all_csi_est, key=lambda z: z[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda z: z[0])]
    all_meta = [x[1] for x in sorted(all_meta, key=lambda z: z[0])]
    
    # Convert to arrays
    dataset = {
        'tx_grids': np.array(all_tx_grids),
        'rx_grids': np.array(all_rx_grids),
        'csi_est': np.array(all_csi_est),
        'labels': np.array(all_labels),
        'meta': all_meta
    }
    
    print(f"✅ Phase 1 dataset generated: {len(dataset['labels'])} samples")
    return dataset

