#!/usr/bin/env python3
"""
🔧 Phase 1: Enhanced Covert Injection with Pattern Variety
==========================================================
Supports fixed and random patterns with configurable subbands.
"""

import numpy as np
import tensorflow as tf
from config.settings import ABLATION_CONFIG


def inject_covert_channel_fixed_phase1(ofdm_np, resource_grid, 
                                       pattern='fixed', 
                                       subband_mode='mid',
                                       selected_subcarriers=None,
                                       selected_symbols=None,
                                       covert_amp=0.5,
                                       power_preserving=True):
    """
    Phase 1: Enhanced injection with pattern and subband variety.
    
    Args:
        ofdm_np: OFDM grid (numpy array, shape: [batch, num_rx, num_tx, symbols, subcarriers])
        resource_grid: Resource grid object
        pattern: 'fixed' or 'random' (symbol/subcarrier selection)
        subband_mode: 'mid' (24-39) or 'random16' (random 16 contiguous subcarriers)
        selected_subcarriers: Pre-selected subcarriers (if None, auto-select based on subband_mode)
        selected_symbols: Pre-selected symbols (if None, auto-select based on pattern)
        covert_amp: Covert amplitude
        power_preserving: Whether to preserve total power
    
    Returns:
        ofdm_np: Modified OFDM grid
        injection_info: Dict with injection metadata
    """
    batch_size, num_rx, num_tx, num_symbols, num_subcarriers = ofdm_np.shape
    
    # Get effective subcarriers (exclude DC/null)
    n_subs = resource_grid.num_effective_subcarriers
    n_syms = num_symbols
    
    # ===== Select Subcarriers =====
    if selected_subcarriers is None:
        if subband_mode == 'mid':
            # Middle band: subcarriers 24-39 (16 subcarriers)
            selected_subcarriers = np.arange(24, min(40, n_subs))
        elif subband_mode == 'random16':
            # Random 16 contiguous subcarriers
            max_start = max(0, n_subs - 16)
            start_idx = np.random.randint(0, max_start + 1)
            selected_subcarriers = np.arange(start_idx, min(start_idx + 16, n_subs))
        else:
            # Default: middle band
            selected_subcarriers = np.arange(24, min(40, n_subs))
    else:
        selected_subcarriers = np.array(selected_subcarriers)
    
    num_covert_subs = len(selected_subcarriers)
    
    # ===== Select Symbols =====
    if selected_symbols is None:
        if pattern == 'fixed':
            # Fixed pattern: symbols 0-3 (first 4 symbols)
            selected_symbols = np.arange(0, min(4, n_syms))
        elif pattern == 'random':
            # Random pattern: 4 random symbols
            num_covert_syms = min(4, n_syms)
            selected_symbols = np.random.choice(n_syms, num_covert_syms, replace=False)
            selected_symbols = np.sort(selected_symbols)
        else:
            # Default: fixed
            selected_symbols = np.arange(0, min(4, n_syms))
    else:
        selected_symbols = np.array(selected_symbols)
    
    num_covert_syms = len(selected_symbols)
    
    # ===== Generate Covert Symbols =====
    # Generate random QPSK symbols for covert channel
    covert_bits = np.random.randint(0, 2, size=(num_covert_syms, num_covert_subs, 2))
    # Map to QPSK symbols
    qpsk_map = {0: 1+1j, 1: -1+1j, 2: 1-1j, 3: -1-1j}
    covert_symbols = np.zeros((num_covert_syms, num_covert_subs), dtype=np.complex64)
    for s in range(num_covert_syms):
        for sc in range(num_covert_subs):
            bit_pair = covert_bits[s, sc, 0] * 2 + covert_bits[s, sc, 1]
            covert_symbols[s, sc] = qpsk_map[bit_pair] * covert_amp
    
    # ===== Inject Covert Signal =====
    orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
    
    # Phase-aligned additive injection (maximizes magnitude boost)
    for s_idx, s in enumerate(selected_symbols):
        for sc_idx, sc in enumerate(selected_subcarriers):
            original = ofdm_np[0, 0, 0, s, sc]
            covert = covert_symbols[s_idx, sc_idx]
            
            # Phase-aligned additive injection
            orig_phase = np.angle(original)
            orig_mag = np.abs(original)
            covert_mag = np.abs(covert)
            
            # Align phases: covert_phase = orig_phase (constructive interference)
            covert_aligned = covert_mag * np.exp(1j * orig_phase)
            
            # Additive injection with phase alignment
            ofdm_np[0, 0, 0, s, sc] = original + covert_aligned
    
    after_inject_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
    power_diff_pct = abs(after_inject_power - orig_power) / (orig_power + 1e-12) * 100.0
    
    # Power preservation (if enabled)
    if power_preserving:
        scale = np.sqrt(orig_power / (after_inject_power + 1e-12))
        ofdm_np[0, 0, 0, :, :] *= scale
        after_inject_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        power_diff_pct = abs(after_inject_power - orig_power) / (orig_power + 1e-12) * 100.0
    
    # ===== Injection Info =====
    injection_info = {
        'pattern': pattern,
        'subband_mode': subband_mode,
        'selected_subcarriers': selected_subcarriers.tolist(),
        'selected_symbols': selected_symbols.tolist(),
        'num_covert_subs': num_covert_subs,
        'num_covert_syms': num_covert_syms,
        'covert_amp': covert_amp,
        'power_preserving': power_preserving,
        'orig_power': float(orig_power),
        'after_inject_power': float(after_inject_power),
        'power_diff_pct': float(power_diff_pct)
    }
    
    return ofdm_np, injection_info

