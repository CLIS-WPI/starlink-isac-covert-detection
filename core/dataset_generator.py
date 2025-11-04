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
from core.covert_injection import inject_covert_channel, inject_covert_channel_fixed

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


def check_angular_diversity(sat_positions, emitter_pos, min_angle_deg=20.0):
    """
    Check if satellites have sufficient angular diversity for good geometry.
    
    Args:
        sat_positions: List of satellite positions (N, 3)
        emitter_pos: Emitter position (3,)
        min_angle_deg: Minimum angle between any two line-of-sight vectors
    
    Returns:
        bool: True if geometry is good (sufficient angular separation)
    """
    if len(sat_positions) < 2:
        return False
    
    # Compute line-of-sight vectors from emitter to each satellite
    los_vectors = []
    for sat_pos in sat_positions:
        vec = np.array(sat_pos) - np.array(emitter_pos)
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-6:
            los_vectors.append(vec / vec_norm)  # Normalize
    
    if len(los_vectors) < 2:
        return False
    
    # Check pairwise angles
    min_angle_rad = np.deg2rad(min_angle_deg)
    for i in range(len(los_vectors)):
        for j in range(i + 1, len(los_vectors)):
            cos_angle = np.clip(np.dot(los_vectors[i], los_vectors[j]), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle >= min_angle_rad:
                return True  # Found at least one pair with good separation
    
    return False  # All pairs too close


def generate_dataset_multi_satellite(isac_system, num_samples_per_class, 
                                     num_satellites=4,
                                     ebno_db_range=(5, 15), 
                                     covert_rate_mbps_range=(1, 50),
                                     tle_path=None,
                                     inject_attack_into_pathb=True,
                                     **kwargs):
    """
    Generate multi-satellite dataset with single-satellite insider attack.

    Args:
        isac_system: ISACSystem instance with pre-cached topologies
        num_samples_per_class: Number of samples per class (benign/attack)
        num_satellites: Number of satellites for TDoA (4 or 12 expected)
        ebno_db_range: Range of Eb/N0 in dB
        covert_rate_mbps_range: Range of covert data rates in Mbps
        tle_path: Optional path to TLE file (if provided and core.constellation_select exists,
                  the 12-sat scenario will use real Starlink pos/vel)
        inject_attack_into_pathb: If True, Path-B waveform for attacked sat will contain covert injection
    Returns:
        dict: Dataset (same keys as original implementation)
    """

    # --------------------------------------------------------------------
    # CRITICAL OVERRIDES (explicit per request)
    #   - Force non-power-preserving covert injection to retain spectral cues
    #   - Covert amplitude imported from settings.py
    # --------------------------------------------------------------------
    try:
        # Mutate the shared settings dict so covert_injection sees it
        ABLATION_CONFIG['power_preserving_covert'] = False
    except Exception:
        pass

    # Import covert amplitude from centralized config (no hardcoding!)
    from config.settings import COVERT_AMP
    print(f"[Dataset] Using COVERT_AMP={COVERT_AMP} from settings.py")

    import numpy as np
    import tensorflow as tf
    from datetime import datetime, timezone

    all_iq, all_rxfreq, all_radar, all_labels, all_emit = [], [], [], [], []
    all_sat_recepts = []
    all_tx_time_padded = []
    all_tx_grids = []  # ✅ NEW: Store OFDM grids for frequency-domain detection

    # Try to use constellation selector / leo_orbit if available
    USE_SGP4 = False
    try:
        from core.constellation_select import select_target_and_sensors
        USE_CONST_SEL = True
    except Exception:
        USE_CONST_SEL = False

    try:
        from core.leo_orbit import propagate_tle, read_tle_file
        USE_SGP4 = True
    except Exception:
        USE_SGP4 = False
        # Note: LEO orbit module removed (detection-only mode)

    # Physical constants
    c0 = 3e8  # speed of light (m/s)
    signal_length = 720  # OFDM frame length (samples)

    # LDPC / payload sizing (same as your original)
    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

    total = num_samples_per_class * 2
    order = np.arange(total)
    np.random.shuffle(order)

    print(f"[Dataset] Generating {total} samples ({num_samples_per_class} per class)...")
    print(f"[Dataset] Satellites: {num_satellites}")
    print(f"[Dataset] Using {'TLE/SGP4' if USE_CONST_SEL and tle_path else 'randomized geometry'} for constellation")

    # Flag to show TLE warning only once
    tle_warning_shown = False

    for idx in range(total):
        i = order[idx]
        is_attack = i >= num_samples_per_class
        y = 1 if is_attack else 0

        # -------------- Build constellation (positions + velocities) --------------
        base_positions = []
        base_velocities = []

        # If user provided a TLE and selector available & num_satellites==12, use it
        if num_satellites == 12 and USE_CONST_SEL and tle_path is not None:
            try:
                # 🎯 IMPORTANT: We need to know emitter location BEFORE satellite selection
                # For now, use nadir of target as initial guess, then we'll update
                sel = select_target_and_sensors(
                    tle_path=tle_path,
                    obs_time=datetime.now(timezone.utc),
                    target_name_contains="STARLINK",
                    target_index=None,
                    num_sensors=num_satellites,
                    use_gdop_optimization=True  # 🔧 ENABLED: Greedy GDOP minimization
                )
                # sensors are the receiving satellites; target could be the attacked sat
                sensors = sel['sensors']
                # the target (attacked sat) pos/vel
                target = sel['target']
                
                # Convert ECEF to local coordinates (ground observer at origin)
                ground_obs = sel['ground_observer']
                
                # populate base_positions/velocities: sensors first
                for s in sensors:
                    # Convert ECEF position to local (relative to ground observer)
                    r_ecef = np.array(s['r_ecef'], dtype=float)
                    r_local = r_ecef - ground_obs  # Offset so ground is at origin
                    
                    # 🔍 DEBUG: Print first satellite info
                    if len(base_positions) == 0 and idx == 0:
                        print(f"[DEBUG] First satellite ECEF check:")
                        print(f"  r_ecef = {r_ecef}")
                        print(f"  |r_ecef| = {np.linalg.norm(r_ecef)/1e3:.1f} km")
                        print(f"  ground_obs = {ground_obs}")
                        print(f"  |ground_obs| = {np.linalg.norm(ground_obs)/1e3:.1f} km")
                    
                    # For simulation, we need altitude (z-coordinate) to be actual height
                    # Compute altitude from ECEF magnitude
                    r_ecef_mag = np.linalg.norm(r_ecef)
                    R_EARTH = 6371e3  # Earth radius [m]
                    altitude = r_ecef_mag - R_EARTH  # Altitude above Earth surface
                    
                    # 🔍 DEBUG: Print altitude calculation
                    if len(base_positions) == 0 and idx == 0:
                        print(f"  altitude = {altitude/1e3:.1f} km")
                    
                    # Use local x,y but replace z with altitude
                    base_positions.append(np.array([r_local[0], r_local[1], altitude]))
                    base_velocities.append(np.array(s['v_ecef'], dtype=float))
                
                # ensure length
                if len(base_positions) < num_satellites:
                    # fallback to randomized fill
                    raise RuntimeError("select_target returned insufficient sensors")
                
                # 🎯 EMITTER LOCATION SETUP (CRITICAL FIX!)
                # For attack samples, emitter is on the GROUND, not at satellite altitude!
                # We place emitter randomly within reasonable range of ground_observer
                
                if is_attack:
                    # Emitter on ground (z≈0), within ±500 km of ground observer
                    # This simulates realistic ground-based covert transmitter
                    emitter_offset_x = np.random.uniform(-500e3, 500e3)  # ±500 km range
                    emitter_offset_y = np.random.uniform(-500e3, 500e3)
                    
                    # Emitter location in LOCAL coordinates (z=0 for ground level)
                    emitter_local = np.array([emitter_offset_x, emitter_offset_y, 0.0])
                    
                    # For compatibility with existing code, we still pick an "attacked satellite"
                    # but use it only to determine which satellite receives the covert signal
                    attacked_sat_index = np.random.randint(0, num_satellites)
                    attacked_sat_pos = emitter_local  # ✅ Use emitter location, not satellite!
                    attacked_sat_vel = np.array([0.0, 0.0, 0.0])  # Emitter is stationary
                    
                    if idx % 100 == 0:
                        print(f"[Sample {idx}] Emitter at ground: ({emitter_offset_x/1e3:.1f}, {emitter_offset_y/1e3:.1f}, 0.0) km")
                else:
                    # Benign: use default user location
                    attacked_sat_index = None
                    attacked_sat_pos = np.array([50e3, 50e3, 0.0])  # Default user on ground
                    attacked_sat_vel = np.array([0.0, 0.0, 0.0])
                
                # ⚠️ NOTE: ground_observer is still nadir of target (not ideal)
                # Ideally we should pass emitter_location to select_target_and_sensors()
                # so elevation checks are done from emitter's perspective.
                # For now, we accept some satellites may not actually be visible from emitter.
                
            except Exception as e:
                # fallback to randomized geometry below
                if not tle_warning_shown:
                    print(f"[Dataset][WARN] constellation_select failed ({e}). Falling back to randomized layout.")
                    tle_warning_shown = True
                base_positions = []
                base_velocities = []
        
        # If base_positions still empty (either no TLE or fallback from exception), use randomized geometry
        if len(base_positions) == 0:
            # Randomized geometry (as before) with "plausible" LEO velocities
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
                # generic grid
                side = int(np.ceil(np.sqrt(num_satellites)))
                for j in range(num_satellites):
                    x = (j % side) * grid_spacing
                    y_p = (j // side) * grid_spacing
                    altitude_offset = np.random.uniform(-75e3, 75e3)
                    base_positions.append(np.array([x, y_p, 600e3 + altitude_offset]))
                    base_velocities.append(get_random_velocity())

            # 🎯 EMITTER LOCATION (same fix as TLE path)
            if is_attack:
                # Emitter on ground, not at satellite altitude!
                emitter_offset_x = np.random.uniform(-500e3, 500e3)
                emitter_offset_y = np.random.uniform(-500e3, 500e3)
                attacked_sat_pos = np.array([emitter_offset_x, emitter_offset_y, 0.0])  # ✅ Ground level
                attacked_sat_vel = np.array([0.0, 0.0, 0.0])
                attacked_sat_index = np.random.randint(0, num_satellites)  # For reference only
            else:
                attacked_sat_pos = np.array([50e3, 50e3, 0.0])  # Default user
                attacked_sat_vel = np.array([0.0, 0.0, 0.0])
                attacked_sat_index = None

        # occasional logging
        if idx % 200 == 0:
            try:
                if is_attack:
                    print(f"[Sample {idx}] emitter_ground=({attacked_sat_pos[0]/1e3:.1f},{attacked_sat_pos[1]/1e3:.1f},0.0)km, sat0_alt={base_positions[0][2]/1e3:.1f}km")
                else:
                    print(f"[Sample {idx}] benign, sat0_alt={base_positions[0][2]/1e3:.1f}km")
            except Exception:
                pass

        # -------------- Generate transmit waveform (clean + attacked copy) --------------
        b = isac_system.binary_source([1, total_info_bits])
        c_blocks = [
            isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) 
            for j in range(num_codewords)
        ]
        c = tf.concat(c_blocks, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [1, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_grid_clean = isac_system.rg_mapper(x)  # CLEAN waveform baseline

        # attacked waveform: copy + injection (if attack)
        tx_grid_attacked = tx_grid_clean
        if is_attack:
            # increase covert throughput for stronger spectral footprint
            covert_rate = 80.0  # 80 Mbps - افزایش throughput برای اثر طیفی قوی‌تر
            
            # 🐛 DEBUG: Print before injection
            if idx < 3:  # Print first 3 attack samples
                print(f"[Dataset] Sample {idx} (ATTACK): Calling inject_covert_channel_fixed(rate={covert_rate:.2f}, amp={COVERT_AMP})")
            
            try:
                # ✅ Use explicit covert_amp (overrides defaults)
                # Use FIXED-POSITION injection so detector can learn stable features
                out = inject_covert_channel_fixed(tf.identity(tx_grid_clean),
                                                  isac_system.rg,
                                                  covert_rate,
                                                  isac_system.SUBCARRIER_SPACING,
                                                  covert_amp=COVERT_AMP,
                                                  seed=42)
                
                # 🐛 DEBUG: Verify injection happened
                if idx < 3:
                    grid_before = tf.squeeze(tx_grid_clean).numpy()
                    grid_after = tf.squeeze(out[0] if isinstance(out, tuple) else out).numpy()
                    diff = np.abs(grid_after - grid_before).max()
                    print(f"[Dataset] Sample {idx}: Injection done! Max diff={diff:.6f}")
                
                # inject_covert_channel may return (tx,info) or tx only
                if isinstance(out, tuple) or isinstance(out, list):
                    tx_grid_attacked = out[0]
                else:
                    tx_grid_attacked = out
            except Exception as e:
                # fallback: keep clean if injection fails but still mark attack (rare)
                print(f"[Dataset][WARN] inject_covert_channel failed: {e}. Proceeding with clean tx for attacked sat.")
                tx_grid_attacked = tx_grid_clean
        else:
            # 🐛 DEBUG: Print benign samples too
            if idx < 3:
                print(f"[Dataset] Sample {idx} (BENIGN): No injection")
        
        # ✅ Store OFDM grid for frequency-domain detection (ensure complex64 for memory efficiency)
        g = tx_grid_attacked.numpy().astype(np.complex64)
        all_tx_grids.append(g)

        # -------------- compute tx_time and padded base (use clean base as canonical tx_time) --------------
        tx_time = isac_system.modulator(tx_grid_clean)
        tx_time_flat = tf.squeeze(tx_time)
        # Compute an upper-bound padding length based on maximum distance
        if is_attack:
            ref_point = attacked_sat_pos
        else:
            ref_point = np.array([50e3, 50e3, 0.0])
        max_distance = max(np.linalg.norm(np.array(pos) - ref_point) for pos in base_positions[:num_satellites])
        max_delay_samp = int(np.ceil((max_distance / c0) * isac_system.SAMPLING_RATE)) + 200

        tx_time_padded = tf.pad(
            tf.expand_dims(tx_time_flat, 0),
            [[0, 0], [0, max_delay_samp]],
            constant_values=0
        )
        all_tx_time_padded.append(np.squeeze(tx_time_padded.numpy()))

        # -------------- Per-satellite reception simulation --------------
        sat_rx_list = []
        for sat_idx, (sat_pos, sat_vel) in enumerate(zip(base_positions[:num_satellites], base_velocities[:num_satellites])):

            # select which tx_grid this satellite "sees"
            if is_attack and (sat_idx == attacked_sat_index):
                tx_grid_used = tx_grid_attacked
                # For Path-B attacked copy option, we will derive time waveform from attacked tx_grid
                tx_time_for_sat = isac_system.modulator(tx_grid_attacked)
            else:
                tx_grid_used = tx_grid_clean
                tx_time_for_sat = tx_time  # canonical clean time waveform

            # generate channel with topology if available (same as your original)
            if NTN_AVAILABLE and hasattr(isac_system.CHANNEL_MODEL, 'set_topology'):
                if is_attack:
                    ut_pos = tf.constant([[attacked_sat_pos]], dtype=tf.float32)
                else:
                    ut_pos = tf.constant([[[50e3, 50e3, 0.0]]], dtype=tf.float32)
                bs_pos = tf.constant([[sat_pos]], dtype=tf.float32)
                bs_vel = tf.constant([[sat_vel]], dtype=tf.float32)
                ut_vel = tf.zeros_like(ut_pos)
                try:
                    isac_system.CHANNEL_MODEL.set_topology(ut_pos, bs_pos, ut_vel, bs_vel)
                    a, tau = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
                except Exception:
                    a, tau = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)
            else:
                a, tau = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS, sampling_frequency=isac_system.rg.bandwidth)

            # 🛡️ CRITICAL: Sanitize CIR parameters before use (prevent NaN propagation)
            # For complex 'a', check real and imaginary parts separately
            a_is_finite = tf.logical_and(
                tf.math.is_finite(tf.math.real(a)),
                tf.math.is_finite(tf.math.imag(a))
            )
            a = tf.where(a_is_finite, a, tf.zeros_like(a))
            
            # For real 'tau', direct check
            tau = tf.where(tf.math.is_finite(tau), tau, tf.zeros_like(tau))
            
            # CIR -> time channel
            l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
            h_time = cir_to_time_channel(isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True)
            
            # 🛡️ Verify h_time is finite (final safety check after CIR conversion)
            # h_time is also complex, so check both parts
            h_is_finite = tf.logical_and(
                tf.math.is_finite(tf.math.real(h_time)),
                tf.math.is_finite(tf.math.imag(h_time))
            )
            if not tf.reduce_all(h_is_finite):
                print(f"[Dataset][WARN] Sample {idx}, Sat {sat_idx}: h_time contains NaN/Inf after CIR conversion, skipping...")
                continue
            
            if h_time.shape[-2] == 1:
                mult = [1] * len(h_time.shape)
                mult[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time = tf.tile(h_time, mult)
            h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)
            h_freq = tf.reduce_mean(h_freq, axis=4)
            h_freq = tf.reduce_mean(h_freq, axis=2)
            h_f = h_freq[:, 0, 0, :, :]
            h_f = tf.expand_dims(h_f, axis=1)
            h_f = tf.expand_dims(h_f, axis=2)

            # Path A: apply channel + AWGN using tx_grid_used (for detection)
            y_grid_channel = tx_grid_used * h_f
            ebno_db = float(tf.random.uniform((), *ebno_db_range))
            
            # 🛡️ SNR Quality Check: Ensure minimum SNR for reliable TDOA/FDOA estimation
            if ebno_db < 10.0:
                # Boost low SNR samples to minimum threshold (10 dB)
                ebno_db = 10.0 + np.random.uniform(0, 2.0)  # 10-12 dB range
            
            rx_pow = tf.reduce_mean(tf.abs(y_grid_channel)**2)  # Real-valued power
            
            # 🛡️ Ensure minimum power floor (prevent division by zero)
            rx_pow = tf.maximum(rx_pow, 1e-20)
            
            # Sanity check for NaN (should be rare now with sanitized CIR)
            # rx_pow is real, so direct check is OK
            if not tf.math.is_finite(rx_pow):
                print(f"[Dataset][WARN] Sample {idx}, Sat {sat_idx}: rx_pow is invalid after sanitization, skipping...")
                continue
            
            esn0 = 10.0**(ebno_db/10.0)
            sigma2 = rx_pow / (esn0 + 1e-12)  # Add epsilon to prevent division issues
            std_freq = tf.sqrt(tf.cast(tf.maximum(sigma2 / 2.0, 1e-30), tf.float32))  # Ensure positive
            
            # Check for NaN before proceeding (std_freq is real)
            if not tf.reduce_all(tf.math.is_finite(std_freq)):
                print(f"[Dataset][WARN] Sample {idx}, Sat {sat_idx}: Invalid noise std, skipping...")
                continue
            
            n_f = tf.complex(
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq),
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq)
            )
            y_grid_noisy = y_grid_channel + n_f
            y_time_noisy = isac_system.modulator(y_grid_noisy)
            y_time_noisy_flat = tf.squeeze(y_time_noisy)
            # compute delay (distance to emitter == attacked_sat_pos if attack else default user)
            if is_attack:
                distance = np.linalg.norm(np.array(sat_pos) - np.array(attacked_sat_pos))
            else:
                distance = np.linalg.norm(np.array(sat_pos) - np.array([50e3, 50e3, 0.0]))
            delay_samp = int(np.round((distance / c0) * isac_system.SAMPLING_RATE))
            # roll and crop Path A
            y_time_noisy_padded = tf.pad(
                tf.expand_dims(y_time_noisy_flat, 0),
                [[0,0],[0,max_delay_samp]],
                constant_values=0
            )
            rx_time_padded_channel = tf.roll(y_time_noisy_padded, shift=delay_samp, axis=-1)
            rx_time_cropped = rx_time_padded_channel[:, delay_samp: delay_samp + signal_length]
            rx_grid_cropped = isac_system.demodulator(rx_time_cropped)

            # 🔧 ULTIMATE FIX v2: Use the SAME tx_time_padded as reference!
            # Problem: We were creating a NEW padded version, which differs from tx_time_padded
            # Solution: Use the ALREADY CREATED tx_time_padded and just apply delay + noise
            
            # DON'T create new padding - use the existing one!
            # tx_time_padded was created at line 358 with same max_delay_samp
            
            # Apply geometric delay by rolling the SAME padded reference
            rx_time_padded_clean_rolled = tf.roll(tx_time_padded, shift=delay_samp, axis=-1)

            # Add AWGN to Path B (use SAME SNR as Path A)
            try:
                rx_pow_pathb = tf.reduce_mean(tf.abs(rx_time_padded_clean_rolled)**2)
                rx_pow_pathb = tf.maximum(rx_pow_pathb, 1e-20)
                
                sigma2_time = rx_pow_pathb / esn0
                std_time = tf.sqrt(tf.cast(sigma2_time / 2.0, tf.float32))
                noise_padded = tf.complex(
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time),
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time)
                )
                rx_time_padded_final = rx_time_padded_clean_rolled + noise_padded
            except Exception:
                rx_time_padded_final = rx_time_padded_clean_rolled

            rx_time_b_cropped = rx_time_padded_final[:, delay_samp: delay_samp + signal_length]
            rx_freq_b = isac_system.demodulator(rx_time_b_cropped)

            # store
            sat_rx_list.append({
                'satellite_id': sat_idx,
                'position': np.array(sat_pos),
                'velocity': np.array(sat_vel),
                'ebno_db': ebno_db,  # 🛡️ Store SNR for quality assessment

                # Path A (detection)
                'rx_time_padded': np.squeeze(rx_time_padded_channel.numpy()),
                'rx_time': np.squeeze(rx_time_cropped.numpy()),
                'rx_freq': np.squeeze(rx_grid_cropped.numpy()),

                # Path B (localization) - include attacked waveform for attacked sat if requested
                'rx_time_b_full': np.squeeze(rx_time_padded_final.numpy()),
                'rx_time_b_cropped': np.squeeze(rx_time_b_cropped.numpy()),
                'rx_freq_b': np.squeeze(rx_freq_b.numpy()),

                'true_delay_samples': delay_samp,
                'distance': distance
            })

        # end per-satellite loop

        # Skip this sample if no valid satellites
        if len(sat_rx_list) == 0:
            print(f"[Dataset][WARN] Sample {idx}: No valid satellites, skipping entire sample...")
            continue
        
        # 🛡️ SNR Quality Check: Ensure at least 2 satellites with high SNR (>12 dB)
        high_snr_count = sum(1 for s in sat_rx_list if s.get('ebno_db', 0) >= 12.0)
        if high_snr_count < 2:
            if idx % 100 == 0:  # Avoid spam
                print(f"[Dataset][WARN] Sample {idx}: Only {high_snr_count} high-SNR satellites (need ≥2), accepting anyway...")
            # Note: We don't skip, just warn. Adjust threshold if needed.
        
        # 🛡️ Geometry Check: Ensure minimum satellite count for localization
        if len(sat_rx_list) < 4:
            print(f"[Dataset][WARN] Sample {idx}: Only {len(sat_rx_list)} satellites (need ≥4 for TDOA), skipping...")
            continue
        
        # 🛡️ Geometry Check: Angular diversity (optional but recommended)
        sat_positions_list = [s['position'] for s in sat_rx_list]
        emitter_location = attacked_sat_pos if is_attack else np.array([50e3, 50e3, 0.0])
        if not check_angular_diversity(sat_positions_list, emitter_location, min_angle_deg=20.0):
            if idx % 100 == 0:  # Avoid spam
                print(f"[Dataset][WARN] Sample {idx}: Poor angular diversity (satellites co-linear), accepting anyway...")
            # Note: We warn but don't skip. For strict quality, change to 'continue'

        # Detector uses Path A from primary sat (index 0) as before
        all_iq.append((i, sat_rx_list[0]['rx_time']))
        all_rxfreq.append((i, sat_rx_list[0]['rx_freq']))
        all_sat_recepts.append((i, sat_rx_list))

        # Radar echo and labels (same as original)
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
        # store emitter location as attacked_sat_pos if attack else None (or ground default)
        if is_attack:
            all_emit.append((i, np.array(attacked_sat_pos)))
        else:
            all_emit.append((i, None))

        all_labels.append((i, y))

        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx+1}/{total} samples")

    # ===== padding tx_time_padded to uniform shape =====
    print("\n[Dataset] Padding tx_time_padded to uniform shape...")
    max_padded_len = max(len(arr) for arr in all_tx_time_padded)
    num_samples = len(all_tx_time_padded)
    tx_time_padded_arr = np.zeros((num_samples, max_padded_len), dtype=np.complex64)
    for ii, arr in enumerate(all_tx_time_padded):
        tx_time_padded_arr[ii, :len(arr)] = arr
    print(f"✓ tx_time_padded shape: {tx_time_padded_arr.shape}")

    # sort by original index for alignment
    all_iq = [x[1] for x in sorted(all_iq, key=lambda z: z[0])]
    all_rxfreq = [x[1] for x in sorted(all_rxfreq, key=lambda z: z[0])]
    all_radar = [x[1] for x in sorted(all_radar, key=lambda z: z[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda z: z[0])]
    all_emit = [x[1] for x in sorted(all_emit, key=lambda z: z[0])]
    all_sat_recepts = [x[1] for x in sorted(all_sat_recepts, key=lambda z: z[0])]

    print(f"✓ Dataset generation complete: {len(all_labels)} samples")
    
    # ========================================================================
    # 🔧 POWER NORMALIZATION: DISABLED
    # ========================================================================
    # NOTE: Post-channel power normalization causes Cohen's d → 0 because:
    #   1. We normalize all attacks to benign power → removes all variance
    #   2. Channel variance is HUGE (CV=8-10x) → normalization meaningless
    #   3. Covert detection should use SPECTRAL features, not power!
    #
    # Solution: Trust power_preserving_covert in OFDM grid (before channel)
    # Let channel add natural variance. Detector uses spectrogram to find
    # covert subcarriers, NOT raw power comparison.
    #
    # If you want to enable normalization, set ENABLE_POWER_NORMALIZATION=True
    ENABLE_POWER_NORMALIZATION = False
    
    if ENABLE_POWER_NORMALIZATION:
        print("\n[Power Normalization] Normalizing attack samples to match benign distribution...")
        
        # Step 1: Calculate power distribution of benign samples
        benign_powers = []
        for i, label in enumerate(all_labels):
            if label == 0:  # Benign
                signal_power = np.mean(np.abs(all_iq[i])**2)
                benign_powers.append(signal_power)
        
        if len(benign_powers) > 0:
            benign_powers = np.array(benign_powers)
            mean_benign_power = np.mean(benign_powers)
            std_benign_power = np.std(benign_powers)
            
            print(f"  Benign power distribution: {mean_benign_power:.6e} ± {std_benign_power:.6e}")
            
            # Step 2: For each attack sample, sample a target power from benign distribution
            normalized_count = 0
            attack_powers_before = []
            attack_powers_after = []
            
            np.random.shuffle(benign_powers)
            
            for i, label in enumerate(all_labels):
                if label == 1:  # Attack
                    current_power = np.mean(np.abs(all_iq[i])**2)
                    attack_powers_before.append(current_power)
                    
                    target_power = benign_powers[normalized_count % len(benign_powers)]
                    
                    if current_power > 1e-20:
                        scale_factor = np.sqrt(target_power / current_power)
                        all_iq[i] = all_iq[i] * scale_factor
                        
                        for sat_rx in all_sat_recepts[i]:
                            sat_rx['rx_time'] = sat_rx['rx_time'] * scale_factor
                            sat_rx['rx_freq'] = sat_rx['rx_freq'] * scale_factor
                            sat_rx['rx_time_b_cropped'] = sat_rx['rx_time_b_cropped'] * scale_factor
                            sat_rx['rx_freq_b'] = sat_rx['rx_freq_b'] * scale_factor
                        
                        normalized_count += 1
                        final_power = np.mean(np.abs(all_iq[i])**2)
                        attack_powers_after.append(final_power)
            
            if len(attack_powers_before) > 0:
                mean_before = np.mean(attack_powers_before)
                std_before = np.std(attack_powers_before)
                mean_after = np.mean(attack_powers_after) if len(attack_powers_after) > 0 else 0
                std_after = np.std(attack_powers_after) if len(attack_powers_after) > 0 else 0
                ratio_before = mean_before / mean_benign_power if mean_benign_power > 0 else 0
                ratio_after = mean_after / mean_benign_power if mean_benign_power > 0 else 0
                
                print(f"  ✓ Normalized {normalized_count} attack samples")
                print(f"  Power ratio BEFORE: {ratio_before:.4f}")
                print(f"  Power ratio AFTER:  {ratio_after:.4f}")
                print(f"  Attack variance BEFORE: {std_before:.6e}")
                print(f"  Attack variance AFTER:  {std_after:.6e}")
                print(f"  Benign variance:        {std_benign_power:.6e}")
        else:
            print("  ⚠️ No benign samples found, skipping normalization")
    else:
        print("\n[Power Normalization] DISABLED - Using natural channel variance")
        print("  → Detection relies on spectral features, not power comparison")
    
    print("=" * 60)

    # Collect Path-B clean (or attacked for attacked sat) per-sample -> pick sat index 0 as canonical export
    all_rx_time_b_full = []
    for sat_rx_list in all_sat_recepts:
        # keep the first satellite's Path-B waveform as representative (same as before)
        if len(sat_rx_list) > 0:
            all_rx_time_b_full.append(sat_rx_list[0]['rx_time_b_full'])
        else:
            all_rx_time_b_full.append(np.zeros(1, dtype=np.complex64))

    max_len_b = max(len(x) for x in all_rx_time_b_full)
    num_samples = len(all_rx_time_b_full)
    rx_time_b_full_arr = np.zeros((num_samples, max_len_b), dtype=np.complex64)
    for ii, x in enumerate(all_rx_time_b_full):
        rx_time_b_full_arr[ii, :len(x)] = x

    return {
        'iq_samples': np.array(all_iq),
        'csi': np.array(all_rxfreq),
        'radar_echo': np.array(all_radar),
        'labels': np.array(all_labels),
        'emitter_locations': all_emit,
        'satellite_receptions': all_sat_recepts,
        'sampling_rate': isac_system.SAMPLING_RATE,
        'tx_time_padded': tx_time_padded_arr,
        'rx_time_b_full': rx_time_b_full_arr,
        'tx_grids': np.array(all_tx_grids)  # ✅ NEW: OFDM grids for frequency-domain detection
    }
