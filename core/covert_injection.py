# ======================================
# 📄 core/covert_injection.py (FIXED with verbose logging)
# ======================================

import numpy as np
import tensorflow as tf
from config.settings import COVERT_AMP, ABLATION_CONFIG
from sionna.phy.mapping import Mapper


def inject_covert_channel(ofdm_frame, resource_grid, covert_rate_mbps,
                          scs, covert_amp=None):
    """
    Inject covert QPSK symbols into the OFDM resource grid for detection.
    
    Args:
        covert_amp: Amplitude scaling (default: use COVERT_AMP from settings.py)
    """
    # Use centralized config if not explicitly provided
    if covert_amp is None:
        covert_amp = COVERT_AMP
    
    # 🐛 DEBUG: Always print when called
    if not hasattr(inject_covert_channel, '_call_count'):
        inject_covert_channel._call_count = 0
    inject_covert_channel._call_count += 1

    if inject_covert_channel._call_count <= 3:  # Print first 3 calls
        print(f"  [Covert] CALL #{inject_covert_channel._call_count}: rate={covert_rate_mbps:.2f} Mbps, amp={covert_amp}")

    if covert_rate_mbps <= 0.0:
        print(f"  [Covert] WARNING: covert_rate_mbps={covert_rate_mbps} <= 0, skipping injection!")
        return ofdm_frame, None

    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = ofdm_frame.shape[-2]

    # Compute number of covert subcarriers from target rate
    symbol_duration = (
        (resource_grid.fft_size + resource_grid.cyclic_prefix_length)
        / (resource_grid.fft_size * scs)
    )
    bits_per_symbol = 2  # QPSK
    symbols_per_second = 1.0 / symbol_duration
    bps_per_sub = bits_per_symbol * symbols_per_second
    base_subs = int((covert_rate_mbps * 1e6) / bps_per_sub)
    base_subs = max(1, min(base_subs, resource_grid.num_effective_subcarriers // 4))

    # Stronger footprint: double subcarriers, up to 50% of spectrum
    strong_subs = min(base_subs * 2, resource_grid.num_effective_subcarriers // 2)

    # Generate random QPSK covert symbols
    covert_bits = tf.random.uniform(
        [batch_size, strong_subs, bits_per_symbol], 0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)  # ✅ No 2.0 multiplication

    # Candidate subcarriers: every other subcarrier
    all_indices = np.arange(resource_grid.num_effective_subcarriers)
    candidates = all_indices[::2]
    if len(candidates) < strong_subs:
        strong_subs = len(candidates)
        covert_syms = covert_syms[:, :strong_subs]
    selected = np.random.choice(candidates, strong_subs, replace=False)

    # Inject into more OFDM symbols for stronger visibility
    L = min(5, num_ofdm_symbols)
    sym_indices = np.random.choice(num_ofdm_symbols, L, replace=False)

    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]

    # ✅ RESPECT ABLATION_CONFIG setting (no hard-coded override!)
    power_preserving = ABLATION_CONFIG.get('power_preserving_covert', False)

    # 🐛 DEBUG: Print first call with effective config
    if not hasattr(inject_covert_channel, '_debug_printed'):
        print(f"  [Covert] Power-preserving: {power_preserving}, covert_amp={covert_amp}")
        inject_covert_channel._debug_printed = True

    # Compute original power if preservation is enabled
    orig_power = None
    if power_preserving:
        try:
            orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)
        except Exception as e:
            print(f"  [Covert] Warning: Could not compute orig_power: {e}")
            orig_power = None

    # Inject by replacing selected subcarriers
    for s in sym_indices:
        for k, sc in enumerate(selected):
            ofdm_np[0, 0, 0, s, sc] = complex(np.asarray(cs[k]).item())

    # Rescale only if power_preserving is enabled and we have a baseline
    if power_preserving and orig_power is not None:
        try:
            new_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)
            scale = np.sqrt(orig_power / (new_power + 1e-12))
            ofdm_np[0, 0, 0, :, :] *= scale
            final_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)
            print(f"  [Covert] Power preserved: {orig_power:.6f} → {new_power:.6f} → {final_power:.6f}")
        except Exception as e:
            print(f"  [Covert] Warning: Power preservation failed: {e}")
    else:
        if inject_covert_channel._call_count <= 3:
            after_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)
            print(f"  [Covert] Power NOT preserved (as intended): power={after_power:.6f}")

    # Random emitter location (for metadata compatibility)
    emitter_location = (
        np.random.uniform(-1000, 1000),
        np.random.uniform(-1000, 1000),
        0.0,
    )

    return tf.convert_to_tensor(ofdm_np), emitter_location


# ======================================
# 🔧 FIXED POSITION COVERT INJECTION
# Inject covert channel at CONSISTENT positions for detectability
# ======================================

def inject_covert_channel_fixed(ofdm_frame, resource_grid, covert_rate_mbps,
                                scs, covert_amp=None, seed=42):
    """
    Inject covert QPSK symbols at RANDOMIZED positions (when enabled in settings).

    Args:
        covert_amp: Amplitude scaling (default: use COVERT_AMP from settings.py)
        seed: Fixed seed for selecting subcarriers/symbols (ignored if randomization enabled)
    Returns: (ofdm_with_covert, emitter_location)
    """
    from config.settings import (
        RANDOMIZE_SUBCARRIERS, 
        RANDOMIZE_SYMBOLS,
        MAX_SUBCARRIERS,
        NUM_INJECT_SYMBOLS
    )
    
    # Use centralized config if not explicitly provided
    if covert_amp is None:
        covert_amp = COVERT_AMP
    
    if covert_rate_mbps <= 0.0:
        return ofdm_frame, None

    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = int(ofdm_frame.shape[-2])

    # Compute number of covert subcarriers from target rate
    symbol_duration = (
        (resource_grid.fft_size + resource_grid.cyclic_prefix_length)
        / (resource_grid.fft_size * scs)
    )
    bits_per_symbol = 2  # QPSK
    symbols_per_second = 1.0 / symbol_duration
    bps_per_sub = bits_per_symbol * symbols_per_second
    n_subs = int((covert_rate_mbps * 1e6) / bps_per_sub)
    n_subs = max(1, min(n_subs, resource_grid.num_effective_subcarriers // 2))

    # ✅ FIXED BAND INJECTION: Always use subcarriers 0-15 for consistent pattern
    if RANDOMIZE_SUBCARRIERS:
        # Limit to first MAX_SUBCARRIERS (e.g., 48 out of 64) for consistent pattern region
        max_sc = min(MAX_SUBCARRIERS, resource_grid.num_effective_subcarriers)
        available_indices = np.arange(max_sc)
        selected_subcarriers = np.random.choice(available_indices, size=min(n_subs, len(available_indices)), replace=False)
    else:
        # 🎯 FIXED: Always use subcarriers 0-15 (or 0 to min(16, n_subs))
        # This creates a consistent spectral pattern that CNN can learn
        num_covert_subs = min(16, n_subs)  # Use first 16 subcarriers
        selected_subcarriers = np.arange(num_covert_subs)

    # ✅ FIXED SYMBOL PATTERN: Always use [1,3,5,7] for consistency
    if RANDOMIZE_SYMBOLS:
        # Always inject into NUM_INJECT_SYMBOLS (e.g., 7) symbols, but vary which ones
        num_to_inject = min(NUM_INJECT_SYMBOLS, num_ofdm_symbols)
        selected_symbols = np.random.choice(num_ofdm_symbols, size=num_to_inject, replace=False).tolist()
    else:
        # 🎯 FIXED: Always use symbols [1, 3, 5, 7] (odd symbols pattern)
        # This creates a consistent temporal pattern that CNN can learn
        fixed_pattern = [1, 3, 5, 7]
        selected_symbols = [s for s in fixed_pattern if s < num_ofdm_symbols]
        # If not enough symbols, fallback to simple pattern
        if len(selected_symbols) < 2:
            selected_symbols = list(range(min(2, num_ofdm_symbols)))

    # Generate covert data (data random, positions fixed)
    covert_bits = tf.random.uniform(
        [batch_size, len(selected_subcarriers), bits_per_symbol], 0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)

    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]

    # 🔧 FIX: Weighted additive injection (preserves power better while keeping pattern detectable)
    # This creates a stronger spectral signature while maintaining power similarity
    for s in selected_symbols:
        for k, sc in enumerate(selected_subcarriers):
            original = ofdm_np[0, 0, 0, s, sc]
            covert = complex(np.asarray(cs[k % cs.shape[0]]).item())
            # Weighted combination: preserves original signal structure + adds covert pattern
            # Alpha controls how much original signal to keep (0.6 = 60% original, 40% covert)
            alpha = 0.6  # Original signal weight
            beta = 0.4   # Covert signal weight (controlled by COVERT_AMP)
            ofdm_np[0, 0, 0, s, sc] = alpha * original + beta * covert

    # 🔍 DEBUG: Print injection details
    mode = "RANDOMIZED" if (RANDOMIZE_SUBCARRIERS or RANDOMIZE_SYMBOLS) else "FIXED"
    if not hasattr(inject_covert_channel_fixed, '_debug_count'):
        inject_covert_channel_fixed._debug_count = 0
    inject_covert_channel_fixed._debug_count += 1
    
    if inject_covert_channel_fixed._debug_count <= 3:  # Print first 3 injections
        print(
            f"  [Covert-{mode}] Sample #{inject_covert_channel_fixed._debug_count}: "
            f"{len(selected_subcarriers)} subcarriers, {len(selected_symbols)} symbols, amp={covert_amp}"
        )
        print(f"  🔍 symbols={selected_symbols}, subcarriers={sorted(selected_subcarriers)[:10]}{'...' if len(selected_subcarriers) > 10 else ''}")

    emitter_location = (
        np.random.uniform(-1000, 1000),
        np.random.uniform(-1000, 1000),
        0.0,
    )

    return tf.convert_to_tensor(ofdm_np), emitter_location


def get_covert_mask(resource_grid, num_covert_subcarriers=10, seed=42):
    """
    Generate a binary mask (Nsym, Nsc) for fixed-position covert injection.
    """
    np.random.seed(seed)

    all_indices = np.arange(resource_grid.num_effective_subcarriers)
    step = max(1, len(all_indices) // (num_covert_subcarriers * 2))
    selected_subcarriers = all_indices[::step][:num_covert_subcarriers]

    n_sym = getattr(resource_grid, 'num_ofdm_symbols', 10)
    if n_sym >= 7:
        selected_symbols = [2, 4, 6]
    elif n_sym >= 3:
        selected_symbols = [1, 2]
    else:
        selected_symbols = [0]

    mask = np.zeros((n_sym, resource_grid.num_effective_subcarriers), dtype=np.int32)
    for s in selected_symbols:
        if s < n_sym:
            mask[s, selected_subcarriers] = 1
    return mask


# ======================================
# 🎯 SEMI-FIXED PATTERN INJECTION
# ======================================

def inject_covert_semi_fixed(ofdm_frame, resource_grid, covert_rate_mbps,
                              scs, covert_amp=None):
    """
    🎯 Semi-fixed pattern injection for better CNN learning.
    
    Strategy:
    - Uses contiguous bands (e.g., 8 consecutive subcarriers)
    - Band starting position chosen from limited options (e.g., 0, 8, 16, 24)
    - Symbol pattern chosen from 2 options (odd vs even symbols)
    - Creates recognizable pattern while maintaining diversity
    
    Args:
        ofdm_frame: OFDM resource grid
        resource_grid: Resource grid configuration
        covert_rate_mbps: Covert channel rate (not used directly in semi-fixed mode)
        scs: Subcarrier spacing
        covert_amp: Amplitude scaling (default: use COVERT_AMP from settings.py)
    
    Returns:
        (ofdm_with_covert, emitter_location)
    """
    from config.settings import (
        BAND_SIZE,
        BAND_START_OPTIONS,
        SYMBOL_PATTERN_OPTIONS,
        NUM_COVERT_SUBCARRIERS
    )
    
    # Use centralized config if not explicitly provided
    if covert_amp is None:
        covert_amp = COVERT_AMP
    
    if covert_rate_mbps <= 0.0:
        return ofdm_frame, None

    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = int(ofdm_frame.shape[-2])

    # 🎯 Semi-fixed subcarrier selection: Contiguous bands
    # Select random starting position from limited options
    band_start = np.random.choice(BAND_START_OPTIONS)
    
    # Create multiple contiguous bands
    num_bands = NUM_COVERT_SUBCARRIERS // BAND_SIZE
    selected_subcarriers = []
    
    for i in range(num_bands):
        band_offset = i * BAND_SIZE
        band_base = (band_start + band_offset) % resource_grid.num_effective_subcarriers
        # Add BAND_SIZE consecutive subcarriers
        for j in range(BAND_SIZE):
            sc = (band_base + j) % resource_grid.num_effective_subcarriers
            if sc < resource_grid.num_effective_subcarriers:
                selected_subcarriers.append(sc)
    
    selected_subcarriers = np.array(selected_subcarriers[:NUM_COVERT_SUBCARRIERS])

    # 🎯 Semi-fixed symbol selection: Choose from pattern options
    symbol_pattern_idx = np.random.choice(len(SYMBOL_PATTERN_OPTIONS))
    selected_symbols = SYMBOL_PATTERN_OPTIONS[symbol_pattern_idx]
    
    # Filter symbols that exist in this OFDM frame
    selected_symbols = [s for s in selected_symbols if s < num_ofdm_symbols]

    # Generate covert QPSK symbols
    bits_per_symbol = 2  # QPSK
    covert_bits = tf.random.uniform(
        [batch_size, len(selected_subcarriers), bits_per_symbol], 0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)

    # Inject into OFDM frame
    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]

    # ✅ Check power preservation setting
    power_preserving = ABLATION_CONFIG.get('power_preserving_covert', False)
    
    # Store original power if needed
    orig_power = None
    if power_preserving:
        orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)

    # 🎯 Weighted combination instead of pure additive
    # This preserves power better while keeping pattern detectable
    alpha = 0.7  # Original signal weight
    beta = 0.3   # Covert signal weight (controlled by COVERT_AMP)
    
    for s in selected_symbols:
        for k, sc in enumerate(selected_subcarriers):
            # Weighted combination: keeps power stable, pattern clear
            original = ofdm_np[0, 0, 0, s, sc]
            covert = complex(np.asarray(cs[k % cs.shape[0]]).item())
            ofdm_np[0, 0, 0, s, sc] = alpha * original + beta * covert
    
    # ✅ Explicit power normalization if enabled
    if power_preserving and orig_power is not None and orig_power > 0:
        current_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :]) ** 2)
        if current_power > 0:
            scale = np.sqrt(orig_power / current_power)
            ofdm_np[0, 0, 0, :, :] *= scale

    # 🔍 DEBUG: Print injection details
    if not hasattr(inject_covert_semi_fixed, '_debug_count'):
        inject_covert_semi_fixed._debug_count = 0
    inject_covert_semi_fixed._debug_count += 1
    
    if inject_covert_semi_fixed._debug_count <= 3:
        print(
            f"  [Covert-SemiFix] Sample #{inject_covert_semi_fixed._debug_count}: "
            f"band_start={band_start}, pattern={symbol_pattern_idx}, amp={covert_amp}"
        )
        print(f"  🎯 symbols={selected_symbols}, subcarriers={sorted(selected_subcarriers)}")

    emitter_location = (
        np.random.uniform(-1000, 1000),
        np.random.uniform(-1000, 1000),
        0.0,
    )

    return tf.convert_to_tensor(ofdm_np), emitter_location