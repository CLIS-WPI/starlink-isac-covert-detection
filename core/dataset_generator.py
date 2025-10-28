# ======================================
# 📄 core/dataset_generator.py
# Purpose: Multi-satellite dataset generation with Path A (detection) and Path B (TDoA)
# OPTIMIZED: Uses topology cache + power preservation + proper logging
# UPDATED: Added satellite velocity generation for FDoA
# FIX:      REVERTED SWAP. Detector trains on Path A (real signal).
# ======================================

import numpy as np
import tensorflow as tf
from config.settings import *
from core.covert_injection import inject_covert_channel

# Try to import NTN utilities
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


def generate_dataset_multi_satellite(isac_system, num_samples_per_class, 
                                     num_satellites=4,
                                     ebno_db_range=(5, 15), 
                                     covert_rate_mbps_range=(1, 50)):
    """
    Generate multi-satellite dataset with Path A (channeled) and Path B (clean for TDoA).
    
    OPTIMIZED:
    - Uses pre-cached topologies (if available) for 500× speedup
    - Returns homogeneous tx_time_padded array (padded to max length)
    - Proper logging for progress tracking
    
    Args:
        isac_system: ISACSystem instance with pre-cached topologies
        num_samples_per_class: Number of samples per class (benign/attack)
        num_satellites: Number of satellites for TDoA
        ebno_db_range: Range of Eb/N0 in dB
        covert_rate_mbps_range: Range of covert data rates in Mbps
    
    Returns:
        dict: Dataset with IQ samples, CSI, labels, emitter locations, and satellite receptions
    """
    all_iq, all_rxfreq, all_radar, all_labels, all_emit = [], [], [], [], []
    all_sat_recepts = []
    all_tx_time_padded = []
    
    # Satellite constellation geometry parameters
    grid_spacing = 100e3  # 100 km spacing for base grid
    
    # LDPC parameters
    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k
    
    # Total samples and shuffle
    total = num_samples_per_class * 2  # benign + attack
    order = np.arange(total)
    np.random.shuffle(order)
    
    # Physical constants
    c0 = 3e8  # speed of light (m/s)
    signal_length = 720  # OFDM frame length (samples)
    
    print(f"[Dataset] Generating {total} samples ({num_samples_per_class} per class)...")
    print(f"[Dataset] Satellites: {num_satellites}")
    # print(f"[Dataset] Using {'cached topologies' if isac_system.topology_cache else 'on-the-fly generation'}")
    print(f"[Dataset] Using on-the-fly topology generation (CRITICAL FIX for valid geometry)")

    
    for idx in range(total):
        i = order[idx]
        is_attack = i >= num_samples_per_class
        y = 1 if is_attack else 0
        
        # ===== Generate per-sample satellite constellation =====
        base_positions = []
        base_velocities = [] # NEW: To store velocities
        
        def get_random_velocity():
            """Generates a random LEO velocity vector."""
            v_mag = 7500.0 + np.random.uniform(-500, 500) # LEO speed ~7.5 km/s
            v_vec = np.random.randn(3) # Random direction
            v_vec = v_vec / np.linalg.norm(v_vec) * v_mag
            return v_vec

        if num_satellites == 4:
            # Simple 2×2 grid with altitude variation
            base_positions_coords = [
                np.array([0.0, 0.0, 600e3]),
                np.array([grid_spacing, 0.0, 600e3]),
                np.array([0.0, grid_spacing, 600e3]),
                np.array([grid_spacing, grid_spacing, 600e3]),
            ]
            # Add random altitude offsets (±75 km)
            for pos in base_positions_coords:
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions.append(np.array([pos[0], pos[1], pos[2] + altitude_offset]))
                base_velocities.append(get_random_velocity()) # NEW
        
        elif num_satellites == 12:
            # 3-ring constellation with altitude diversity (Starlink-like)
            user_center_x, user_center_y = 75e3, 75e3
            shells = [545e3, 575e3, 345e3]  # Shell altitudes
            shell_weights = [0.5, 0.35, 0.15]  # Probability distribution
            
            # Ring 1 (inner, 4 satellites)
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(100e3, 140e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-30e3, 30e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity()) # NEW
            
            # Ring 2 (middle, 4 satellites)
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.pi/4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(220e3, 280e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-40e3, 40e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity()) # NEW
            
            # Ring 3 (outer, 4 satellites)
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(380e3, 450e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-50e3, 50e3)
                base_positions.append(np.array([x, y_p, z]))
                base_velocities.append(get_random_velocity()) # NEW
        
        else:
            # Generic grid layout for other satellite counts
            side = int(np.ceil(np.sqrt(num_satellites)))
            for j in range(num_satellites):
                x = (j % side) * grid_spacing
                y_p = (j // side) * grid_spacing
                # Add altitude variation
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions.append(np.array([x, y_p, 600e3 + altitude_offset]))
                base_velocities.append(get_random_velocity()) # NEW
        
        # Log constellation occasionally (not too verbose)
        if idx % 500 == 0 or idx == 0:
            try:
                altitudes = [f'{p[2]/1e3:.1f}km' for p in base_positions[:min(3, len(base_positions))]]
                print(f"  [Sample {idx}] New constellation generated. Alts: {altitudes}...")
            except Exception:
                pass
        
        # ===== Generate transmit signal =====
        b = isac_system.binary_source([1, total_info_bits])
        c_blocks = [
            isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) 
            for j in range(num_codewords)
        ]
        c = tf.concat(c_blocks, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [1, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_grid = isac_system.rg_mapper(x)
        
        # ===== Covert injection (if attack sample) =====
        emitter_loc = None
        if is_attack:
            covert_rate = np.random.uniform(*covert_rate_mbps_range)
            # Place ground emitter in 200 km × 200 km area
            emitter_loc = np.array([
                np.random.uniform(-50e3, 150e3),
                np.random.uniform(-50e3, 150e3),
                0.0  # ground level
            ])
            tx_grid, _ = inject_covert_channel(
                tx_grid, isac_system.rg, covert_rate, 
                isac_system.SUBCARRIER_SPACING, COVERT_AMP
            )
        
        # ===== Calculate max delay for padding =====
        if emitter_loc is not None:
            ref_point = emitter_loc
        else:
            ref_point = np.array([50e3, 50e3, 0.0])  # Default user location
        
        max_distance = max([
            np.linalg.norm(sat_pos - ref_point) 
            for sat_pos in base_positions[:num_satellites]
        ])
        max_delay_samp = int(np.ceil((max_distance / c0) * isac_system.SAMPLING_RATE)) + 200
        
        # ===== Modulate clean transmit signal (Used as base for Path B) =====
        tx_time = isac_system.modulator(tx_grid)
        tx_time_flat = tf.squeeze(tx_time)
        tx_time_padded = tf.pad(
            tf.expand_dims(tx_time_flat, 0),
            [[0, 0], [0, max_delay_samp]],
            constant_values=0
        )
        
        # Store padded transmit waveform
        all_tx_time_padded.append(np.squeeze(tx_time_padded.numpy()))
        
        # ===== Multi-satellite reception =====
        sat_rx_list = []
        
        # NEW: Iterate over positions and velocities
        for sat_idx, (sat_pos, sat_vel) in enumerate(zip(base_positions[:num_satellites], base_velocities[:num_satellites])):
            
            # ===== Generate channel =====
            
            # 🛑 CRITICAL FIX: DO NOT USE CACHE. 
            # The cache loads a RANDOM topology. We MUST use the
            # sat_pos and sat_vel for THIS satellite.
            #
            if NTN_AVAILABLE and hasattr(isac_system.CHANNEL_MODEL, 'set_topology'):
                
                # We MUST generate the topology on the fly using this sample's
                # specific geometry.
                
                # Emitter (user) position:
                # Use the covert emitter location, or the default user location
                if emitter_loc is not None:
                    ut_pos = tf.constant([[emitter_loc]], dtype=tf.float32)
                else:
                    ut_pos = tf.constant([[[50e3, 50e3, 0.0]]], dtype=tf.float32) # Default user
                
                # Satellite position and velocity
                bs_pos = tf.constant([[sat_pos]], dtype=tf.float32)
                bs_vel = tf.constant([[sat_vel]], dtype=tf.float32)
                ut_vel = tf.zeros_like(ut_pos) # Emitter is stationary
                
                try:
                    # Manually set the topology for the channel model
                    isac_system.CHANNEL_MODEL.set_topology(ut_pos, bs_pos, ut_vel, bs_vel)
                    a, tau = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
                
                except Exception as e:
                    # Fallback if topology fails (e.g., sat is below horizon)
                    # print(f"Warning: Sionna topology failed: {e}. Falling back to Rayleigh.")
                    a, tau = isac_system.CHANNEL_MODEL(
                        1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                        sampling_frequency=isac_system.rg.bandwidth
                    )
            
            else:
                # Rayleigh channel (no topology)
                a, tau = isac_system.CHANNEL_MODEL(
                    1, 
                    num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                    sampling_frequency=isac_system.rg.bandwidth
                )
            
            # Convert CIR to time-domain channel
            l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
            h_time = cir_to_time_channel(
                isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True
            )
            
            # Tile if needed (expand to all OFDM symbols)
            if h_time.shape[-2] == 1:
                mult = [1] * len(h_time.shape)
                mult[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time = tf.tile(h_time, mult)
            
            # Convert to frequency domain
            h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)
            h_freq = tf.reduce_mean(h_freq, axis=4)  # average TX antennas
            h_freq = tf.reduce_mean(h_freq, axis=2)  # average RX antennas
            h_f = h_freq[:, 0, 0, :, :]
            h_f = tf.expand_dims(h_f, axis=1)
            h_f = tf.expand_dims(h_f, axis=2)
            
            # ===== Path A: Apply channel + noise (for Detection) =====
            y_grid_channel = tx_grid * h_f
            
            # Add AWGN in frequency domain
            ebno_db = float(tf.random.uniform((), *ebno_db_range))
            rx_pow = tf.reduce_mean(tf.abs(y_grid_channel)**2)
            esn0 = 10.0**(ebno_db/10.0)
            sigma2 = rx_pow / esn0
            std_freq = tf.sqrt(tf.cast(sigma2 / 2.0, tf.float32))
            
            n_f = tf.complex(
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq),
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq)
            )
            y_grid_noisy = y_grid_channel + n_f
            
            # Modulate to time domain
            y_time_noisy = isac_system.modulator(y_grid_noisy)
            y_time_noisy_flat = tf.squeeze(y_time_noisy)
            y_time_noisy_padded = tf.pad(
                tf.expand_dims(y_time_noisy_flat, 0),
                [[0, 0], [0, max_delay_samp]],
                constant_values=0
            )
            
            # Calculate propagation delay
            if emitter_loc is not None:
                distance = np.linalg.norm(sat_pos - emitter_loc)
            else:
                user_pos = np.array([50e3, 50e3, 0.0])
                distance = np.linalg.norm(sat_pos - user_pos)
            
            delay_samp = int(np.round((distance / c0) * isac_system.SAMPLING_RATE))
            
            # Roll and crop (Path A for detection)
            rx_time_padded_channel = tf.roll(y_time_noisy_padded, shift=delay_samp, axis=-1)
            rx_time_cropped = rx_time_padded_channel[:, delay_samp : delay_samp + signal_length]
            rx_grid_cropped = isac_system.demodulator(rx_time_cropped)
            
            # ===== Path B: Clean geometric delay only (for Localization) =====
            # ❗ No channel effect here — just manual delay + AWGN.
            rx_time_padded_clean_rolled = tf.roll(tx_time_padded, shift=delay_samp, axis=-1)

            # Add AWGN (same SNR as Path A)
            try:
                tx_pow = tf.reduce_mean(tf.abs(tx_time_padded)**2)
                sigma2_time = tx_pow / esn0
                std_time = tf.sqrt(tf.cast(sigma2_time / 2.0, tf.float32))
                noise_padded = tf.complex(
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time),
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time)
                )
                rx_time_padded_final = rx_time_padded_clean_rolled + noise_padded
            except:
                # Fallback if power calculation fails
                rx_time_padded_final = rx_time_padded_clean_rolled

            # --- NEW: Create cropped, demodulated Path B features for detector ---
            rx_time_b_cropped = rx_time_padded_final[:, delay_samp : delay_samp + signal_length]
            rx_freq_b = isac_system.demodulator(rx_time_b_cropped)
            # --- END NEW ---

           
            # Store satellite reception data
            sat_rx_list.append({
                'satellite_id': sat_idx,
                'position': sat_pos,
                'velocity': sat_vel, # NEW: Added velocity
                
                # Path A (Full Channel) - Now used for DETECTOR
                'rx_time_padded': np.squeeze(rx_time_padded_channel.numpy()), # Full delayed Path A
                'rx_time': np.squeeze(rx_time_cropped.numpy()),            # Cropped delayed Path A
                'rx_freq': np.squeeze(rx_grid_cropped.numpy()),            # Cropped demod Path A
                
                # Path B (Clean + Noise) - Now used for LOCALIZATION
                'rx_time_b_full': np.squeeze(rx_time_padded_final.numpy()), # Full delayed Path B
                'rx_time_b_cropped': np.squeeze(rx_time_b_cropped.numpy()), # Cropped delayed Path B
                'rx_freq_b': np.squeeze(rx_freq_b.numpy()),           # Cropped demod Path B

                'true_delay_samples': delay_samp,
                'distance': distance
            })
        
        # Store primary satellite (index 0) for detector features
        # --- FIX: REVERTED. Detector trains on Path A (real signal) ---
        all_iq.append((i, sat_rx_list[0]['rx_time']))
        all_rxfreq.append((i, sat_rx_list[0]['rx_freq']))
        # --- END FIX ---

        all_sat_recepts.append((i, sat_rx_list))
        
        # ===== Radar echo (independent noise) =====
        radar_delay = np.random.randint(50, 150)
        radar_atten = 0.3
        tx_flat = tf.squeeze(tx_time)
        radar_echo = tf.roll(tx_flat, shift=radar_delay, axis=-1) * radar_atten
        radar_std = tf.constant(0.05, dtype=tf.float32)
        radar_echo += tf.complex(
            tf.random.normal(tf.shape(radar_echo), stddev=radar_std),
            tf.random.normal(tf.shape(radar_echo), stddev=radar_std)
        )
        
        all_radar.append((i, np.squeeze(radar_echo.numpy())))
        all_labels.append((i, y))
        all_emit.append((i, emitter_loc))
        
        # Progress logging
        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx+1}/{total} samples")
    
    # ===== Pad all tx_time_padded to uniform shape =====
    print("\n[Dataset] Padding tx_time_padded to uniform shape...")
    
    # Find max length across all samples
    max_padded_len = max(len(arr) for arr in all_tx_time_padded)
    num_samples = len(all_tx_time_padded)
    
    # Allocate 2D array: [num_samples, max_padded_len]
    tx_time_padded_arr = np.zeros((num_samples, max_padded_len), dtype=np.complex64)
    
    # Copy each sample into the left portion (zero-pad the tail)
    for i, arr in enumerate(all_tx_time_padded):
        tx_time_padded_arr[i, :len(arr)] = arr
    
    print(f"✓ tx_time_padded shape: {tx_time_padded_arr.shape}")
    
    # ===== Sort by original index for alignment =====
    all_iq = [x[1] for x in sorted(all_iq, key=lambda z: z[0])]
    all_rxfreq = [x[1] for x in sorted(all_rxfreq, key=lambda z: z[0])]
    all_radar = [x[1] for x in sorted(all_radar, key=lambda z: z[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda z: z[0])]
    all_emit = [x[1] for x in sorted(all_emit, key=lambda z: z[0])]
    all_sat_recepts = [x[1] for x in sorted(all_sat_recepts, key=lambda z: z[0])]
    
    print(f"✓ Dataset generation complete: {len(all_labels)} samples")

    # ===== Collect clean Path-B waveforms (for TDoA) =====
    # ===== Collect and pad clean Path-B waveforms (for TDoA) =====
    all_rx_time_b_full = []
    for sat_rx_list in all_sat_recepts:
        if len(sat_rx_list) > 0:
            all_rx_time_b_full.append(sat_rx_list[0]['rx_time_b_full'])
        else:
            all_rx_time_b_full.append(np.zeros(1, dtype=np.complex64))

    # پیدا کردن طول حداکثر بین تمام سیگنال‌ها
    max_len_b = max(len(x) for x in all_rx_time_b_full)
    num_samples = len(all_rx_time_b_full)

    # پد کردن همه‌ی سیگنال‌ها به طول یکنواخت
    rx_time_b_full_arr = np.zeros((num_samples, max_len_b), dtype=np.complex64)
    for i, x in enumerate(all_rx_time_b_full):
        rx_time_b_full_arr[i, :len(x)] = x

    # ===== Return dataset (Path B padded) =====
    return {
        'iq_samples': np.array(all_iq),
        'csi': np.array(all_rxfreq),
        'radar_echo': np.array(all_radar),
        'labels': np.array(all_labels),
        'emitter_locations': all_emit,
        'satellite_receptions': all_sat_recepts,
        'sampling_rate': isac_system.SAMPLING_RATE,
        'tx_time_padded': tx_time_padded_arr,
        'rx_time_b_full': rx_time_b_full_arr  # ✅ Uniform shape (padded)
    }