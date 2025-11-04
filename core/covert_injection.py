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

    # Generate random QPSK covert symbols with amplitude boost (×2)
    covert_bits = tf.random.uniform(
        [batch_size, strong_subs, bits_per_symbol], 0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp * 2.0, tf.complex64)

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

    # Force power preservation OFF for detection (override config)
    _cfg_power_preserving = ABLATION_CONFIG.get('power_preserving_covert', False)
    power_preserving = False

    # 🐛 DEBUG: Print first call with effective config
    if not hasattr(inject_covert_channel, '_debug_printed'):
        print(f"  [Covert] DEBUG: power_preserving={power_preserving} (cfg={_cfg_power_preserving}), covert_amp={covert_amp}")
        inject_covert_channel._debug_printed = True

    # Optionally compute original power (won't be used when forced False)
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
    Inject covert QPSK symbols at FIXED positions (deterministic by seed).

    Args:
        covert_amp: Amplitude scaling (default: use COVERT_AMP from settings.py)
        seed: Fixed seed for selecting subcarriers/symbols
    Returns: (ofdm_with_covert, emitter_location)
    """
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

    # Fixed subcarrier selection by seed (DENSER selection for stronger footprint)
    np.random.seed(seed)
    all_indices = np.arange(resource_grid.num_effective_subcarriers)

    # choose more subs with denser distribution (×5 for maximum coverage)
    step = max(1, len(all_indices) // (n_subs * 5))  # denser -> more subcarriers injected
    selected_subcarriers = all_indices[::step][:n_subs]

    # Inject across more OFDM symbols (middle region) but keep deterministic
    if num_ofdm_symbols >= 10:
        selected_symbols = list(range(1, min(num_ofdm_symbols-1, 8)))  # up to 7 middle symbols
    elif num_ofdm_symbols >= 7:
        selected_symbols = [1,2,3,4]
    elif num_ofdm_symbols >= 3:
        selected_symbols = [1,2]
    else:
        selected_symbols = [0]

    # Generate covert data (data random, positions fixed)
    covert_bits = tf.random.uniform(
        [batch_size, len(selected_subcarriers), bits_per_symbol], 0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)

    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]

    # covert_syms generated earlier; use additive injection (stronger footprint)
    for s in selected_symbols:
        for k, sc in enumerate(selected_subcarriers):
            ofdm_np[0, 0, 0, s, sc] += complex(np.asarray(cs[k % cs.shape[0]]).item())

    # 🔍 DEBUG مورد 2: چاپ دقیق اندیس‌های تزریق
    print(
        f"  [Covert-Fixed] Injected {len(selected_subcarriers)} subcarriers at symbols {selected_symbols} with amp={covert_amp}"
    )
    print(f"  🔍 DEBUG injection: symbols={selected_symbols}, subcarriers=[{selected_subcarriers[0]}..{selected_subcarriers[-1]}], step={step}")

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