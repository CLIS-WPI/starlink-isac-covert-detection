# ======================================
# 📄 core/covert_injection.py (FIXED with verbose logging)
# ======================================

import numpy as np
import tensorflow as tf
from config.settings import COVERT_AMP, ABLATION_CONFIG
from sionna.phy.mapping import Mapper


def inject_covert_channel(ofdm_frame, resource_grid, covert_rate_mbps, 
                          scs, covert_amp=COVERT_AMP):
    """
    Inject covert QPSK symbols with power preservation.
    
    FIXED: Added verbose logging to match original output.
    """
    if covert_rate_mbps <= 0.0:
        return ofdm_frame, None
    
    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = ofdm_frame.shape[-2]
    
    # Compute number of covert subcarriers
    symbol_duration = (
        (resource_grid.fft_size + resource_grid.cyclic_prefix_length) / 
        (resource_grid.fft_size * scs)
    )
    bits_per_symbol = 2  # QPSK
    symbols_per_second = 1.0 / symbol_duration
    bps_per_sub = bits_per_symbol * symbols_per_second
    num_covert_subcarriers = int((covert_rate_mbps * 1e6) / bps_per_sub)
    num_covert_subcarriers = max(
        1,
        min(num_covert_subcarriers, resource_grid.num_effective_subcarriers // 4)
    )
    
    # Generate random QPSK covert symbols
    covert_bits = tf.random.uniform(
        [batch_size, num_covert_subcarriers, bits_per_symbol],
        0, 2, dtype=tf.int32
    )
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)
    
    # Select sparse subcarriers
    all_indices = np.arange(resource_grid.num_effective_subcarriers)
    candidates = all_indices[::4]
    if len(candidates) < num_covert_subcarriers:
        num_covert_subcarriers = len(candidates)
        covert_syms = covert_syms[:, :num_covert_subcarriers]
    selected = np.random.choice(candidates, num_covert_subcarriers, replace=False)
    
    # Inject into 3 random OFDM symbols
    L = min(3, num_ofdm_symbols)
    sym_indices = np.random.choice(num_ofdm_symbols, L, replace=False)
    
    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]
    
    # ✅ STEP 1: Save original power (with verbose logging)
    orig_power = None
    if ABLATION_CONFIG.get('power_preserving_covert', True):
        try:
            orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        except Exception as e:
            print(f"  [Covert] Warning: Could not compute orig_power: {e}")
            orig_power = None
    
    # STEP 2: Inject covert symbols
    for s in sym_indices:
        for k, sc in enumerate(selected):
            ofdm_np[0, 0, 0, s, sc] += complex(np.asarray(cs[k]).item())
    
    # ✅ STEP 3: Rescale to preserve original power (with verbose logging)
    if orig_power is not None:
        new_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        scale = np.sqrt(orig_power / (new_power + 1e-12))
        ofdm_np[0, 0, 0, :, :] *= scale
        
        # ✅ VERBOSE: Print power preservation (always print, not in try/except)
        final_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        print(f"  [Covert] Power preserved: {orig_power:.6f} → {new_power:.6f} → {final_power:.6f}")
    
    # Random emitter location
    emitter_location = (
        np.random.uniform(-1000, 1000),
        np.random.uniform(-1000, 1000),
        0.0
    )
    
    return tf.convert_to_tensor(ofdm_np), emitter_location