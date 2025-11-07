#!/usr/bin/env python3
"""
🔧 Phase 5: Enhanced CSI Estimation
===================================
Improved CSI estimation methods:
1. LS (Least Squares) with smoothing
2. LMMSE (Linear Minimum Mean Square Error) - simplified
3. NMSE computation for quality assessment
"""

import numpy as np
import tensorflow as tf
from scipy import ndimage


def estimate_csi_ls_smooth(tx_grid, rx_grid, pilot_symbols=[2, 7], smoothing=True, interpolation='nearest'):
    """
    LS estimation with optional 2D smoothing.
    
    Args:
        tx_grid: Transmitted grid (symbols, subcarriers)
        rx_grid: Received grid (symbols, subcarriers)
        pilot_symbols: List of pilot symbol indices
        smoothing: Apply 2D median filter for smoothing
        interpolation: 'nearest' (default), 'linear', or 'spline' for comparison
    
    Returns:
        h_est: Estimated CSI (symbols, subcarriers)
    """
    num_symbols, num_subcarriers = rx_grid.shape[0], rx_grid.shape[1]
    
    # Estimate CSI at pilot symbols
    h_est_pilots = []
    pilot_positions = []
    
    for sym_idx in pilot_symbols:
        if sym_idx < num_symbols:
            for sc_idx in range(num_subcarriers):
                tx_pilot = tx_grid[sym_idx, sc_idx]
                rx_pilot = rx_grid[sym_idx, sc_idx]
                if np.abs(tx_pilot) > 1e-9:
                    h_pilot = rx_pilot / tx_pilot
                    h_est_pilots.append(h_pilot)
                    pilot_positions.append((sym_idx, sc_idx))
    
    # Interpolate to all tones
    h_est = np.zeros((num_symbols, num_subcarriers), dtype=np.complex64)
    
    if len(h_est_pilots) > 0:
        h_est_pilots = np.array(h_est_pilots)
        pilot_positions = np.array(pilot_positions)
        
        # 🔧 FINAL CHECK: Support multiple interpolation methods for comparison
        if interpolation == 'nearest':
            # Nearest neighbor interpolation (default, stable)
            for sym_idx in range(num_symbols):
                for sc_idx in range(num_subcarriers):
                    distances = np.abs(pilot_positions[:, 0] - sym_idx) + np.abs(pilot_positions[:, 1] - sc_idx)
                    nearest_idx = np.argmin(distances)
                    h_est[sym_idx, sc_idx] = h_est_pilots[nearest_idx]
        elif interpolation == 'linear':
            # Linear interpolation per symbol (better for fast-changing channels)
            from scipy.interpolate import griddata
            grid_y, grid_x = np.mgrid[0:num_symbols, 0:num_subcarriers]
            points = pilot_positions
            h_est = griddata(points, h_est_pilots, (grid_y, grid_x), method='linear', fill_value=0.0)
            # Fill NaN with nearest neighbor
            nan_mask = np.isnan(h_est)
            if np.any(nan_mask):
                for sym_idx in range(num_symbols):
                    for sc_idx in range(num_subcarriers):
                        if nan_mask[sym_idx, sc_idx]:
                            distances = np.abs(pilot_positions[:, 0] - sym_idx) + np.abs(pilot_positions[:, 1] - sc_idx)
                            nearest_idx = np.argmin(distances)
                            h_est[sym_idx, sc_idx] = h_est_pilots[nearest_idx]
        else:  # 'spline' or default to nearest
            # Use nearest for now (spline requires more complex setup)
            for sym_idx in range(num_symbols):
                for sc_idx in range(num_subcarriers):
                    distances = np.abs(pilot_positions[:, 0] - sym_idx) + np.abs(pilot_positions[:, 1] - sc_idx)
                    nearest_idx = np.argmin(distances)
                    h_est[sym_idx, sc_idx] = h_est_pilots[nearest_idx]
        
        # Optional: 2D smoothing with median filter
        if smoothing:
            # Apply median filter to magnitude and phase separately
            h_mag = np.abs(h_est)
            h_phase = np.angle(h_est)
            
            # Smooth magnitude
            h_mag_smooth = ndimage.median_filter(h_mag, size=(3, 3))
            
            # Smooth phase (unwrap first)
            h_phase_unwrapped = np.unwrap(h_phase, axis=0)
            h_phase_smooth = ndimage.median_filter(h_phase_unwrapped, size=(3, 3))
            h_phase_smooth = np.angle(np.exp(1j * h_phase_smooth))  # Wrap back
            
            # Reconstruct
            h_est = h_mag_smooth * np.exp(1j * h_phase_smooth)
    else:
        # Fallback: simple division
        denom = np.where(np.abs(tx_grid) > 1e-9, tx_grid, 1e-9)
        h_est = rx_grid / denom
    
    return h_est


def estimate_csi_lmmse_simple(tx_grid, rx_grid, pilot_symbols=[2, 7], snr_db=20.0):
    """
    Simplified LMMSE estimation.
    
    LMMSE: H_LMMSE = R_HH (R_HH + σ²I)^(-1) H_LS
    
    Simplified version:
    - Assume R_HH is diagonal (uncorrelated taps)
    - Use empirical variance from LS estimate
    - Apply Wiener filter in frequency domain
    
    Args:
        tx_grid: Transmitted grid
        rx_grid: Received grid
        pilot_symbols: List of pilot symbol indices
        snr_db: SNR in dB (for noise variance)
    
    Returns:
        h_est: LMMSE estimated CSI
    """
    # First get LS estimate
    h_ls = estimate_csi_ls_smooth(tx_grid, rx_grid, pilot_symbols, smoothing=False)
    
    # Compute noise variance from SNR
    signal_power = np.mean(np.abs(rx_grid)**2)
    snr_linear = 10.0**(snr_db / 10.0)
    noise_variance = signal_power / (snr_linear + 1e-12)
    
    # Estimate channel variance from LS estimate
    h_variance = np.var(h_ls)
    
    # Simplified LMMSE: Wiener filter
    # H_LMMSE = (σ²_H / (σ²_H + σ²_N)) * H_LS
    if h_variance > 0:
        alpha = h_variance / (h_variance + noise_variance)
        h_lmmse = alpha * h_ls
    else:
        h_lmmse = h_ls
    
    # Apply smoothing
    h_mag = np.abs(h_lmmse)
    h_phase = np.angle(h_lmmse)
    
    h_mag_smooth = ndimage.median_filter(h_mag, size=(3, 3))
    h_phase_unwrapped = np.unwrap(h_phase, axis=0)
    h_phase_smooth = ndimage.median_filter(h_phase_unwrapped, size=(3, 3))
    h_phase_smooth = np.angle(np.exp(1j * h_phase_smooth))
    
    h_lmmse = h_mag_smooth * np.exp(1j * h_phase_smooth)
    
    return h_lmmse


def compute_nmse(h_true, h_est):
    """
    Compute Normalized Mean Square Error (NMSE) in dB.
    
    NMSE = 10 * log10(||H_est - H_true||² / ||H_true||²)
    
    Args:
        h_true: True CSI (symbols, subcarriers) or None
        h_est: Estimated CSI (symbols, subcarriers)
    
    Returns:
        nmse_db: NMSE in dB (lower is better)
    """
    if h_true is None:
        return None
    
    # Flatten for computation
    h_true_flat = h_true.flatten()
    h_est_flat = h_est.flatten()
    
    # Compute NMSE
    numerator = np.sum(np.abs(h_est_flat - h_true_flat)**2)
    denominator = np.sum(np.abs(h_true_flat)**2)
    
    if denominator > 1e-12:
        nmse_linear = numerator / denominator
        nmse_db = 10.0 * np.log10(nmse_linear + 1e-12)
    else:
        nmse_db = np.inf
    
    return float(nmse_db)


def compute_csi_quality_metrics(h_est, h_true=None):
    """
    Compute CSI quality metrics.
    
    Args:
        h_est: Estimated CSI
        h_true: True CSI (optional)
    
    Returns:
        dict: Quality metrics (nmse_db, variance, mean_mag, etc.)
    """
    metrics = {}
    
    # NMSE (if true CSI available)
    if h_true is not None:
        metrics['nmse_db'] = compute_nmse(h_true, h_est)
    else:
        metrics['nmse_db'] = None
    
    # Variance (higher variance may indicate noise)
    h_mag = np.abs(h_est)
    metrics['variance'] = float(np.var(h_mag))
    metrics['mean_mag'] = float(np.mean(h_mag))
    metrics['std_mag'] = float(np.std(h_mag))
    
    # Consistency check: variance should be reasonable
    # Very high variance suggests noisy estimate
    if metrics['variance'] > 1.0:
        metrics['quality'] = 'poor'
    elif metrics['variance'] > 0.1:
        metrics['quality'] = 'moderate'
    else:
        metrics['quality'] = 'good'
    
    return metrics


def mmse_equalize(rx_grid, h_est, snr_db=20.0, alpha_reg=None, blend_factor=None):
    """
    MMSE Equalization for OFDM signal.
    
    Formula: X_est = (H* / (|H|² + α)) * Y
    
    Where:
    - H*: conjugate of estimated channel
    - |H|²: magnitude squared of channel
    - α: regularization parameter (noise variance / signal power)
    - Y: received signal
    
    Args:
        rx_grid: Received OFDM grid (symbols, subcarriers) - complex
        h_est: Estimated CSI (symbols, subcarriers) - complex
        snr_db: SNR in dB (for computing α if not provided)
        alpha_reg: Regularization parameter (if None, computed from SNR)
        blend_factor: Blending factor for low SNR (0.0 = raw, 1.0 = equalized)
                     If None, auto-compute based on SNR
    
    Returns:
        x_eq: Equalized OFDM grid (symbols, subcarriers) - complex
        eq_info: Dict with equalization info (alpha_used, snr_eq, etc.)
    """
    # Compute regularization parameter α
    # 🔧 FINAL CHECK: α is defined in power units (noise_variance), global (not per-subcarrier)
    if alpha_reg is None:
        # α = noise_variance (in power units, not dimensionless)
        signal_power = np.mean(np.abs(rx_grid)**2)
        snr_linear = 10.0**(snr_db / 10.0)
        noise_variance = signal_power / (snr_linear + 1e-12)  # Power units
        # Use adaptive α: larger for low SNR to prevent noise amplification
        if snr_db < 3.0:  # Very low SNR
            alpha_reg = noise_variance * 10.0  # More regularization (still in power units)
        elif snr_db < 10.0:  # Low SNR
            alpha_reg = noise_variance * 5.0
        else:  # Normal SNR
            alpha_reg = noise_variance
    else:
        noise_variance = alpha_reg
    
    # MMSE equalization: X_est = (H* / (|H|² + α)) * Y
    h_conj = np.conj(h_est)
    h_mag_sq = np.abs(h_est)**2
    denominator = h_mag_sq + alpha_reg + 1e-12  # Prevent division by zero
    
    # Equalization filter
    eq_filter = h_conj / denominator
    
    # Apply equalization
    x_eq = rx_grid * eq_filter
    
    # Compute pre-equalization SNR (raw signal)
    signal_power_raw = np.mean(np.abs(rx_grid)**2)
    noise_power_raw = noise_variance  # Input noise variance
    snr_raw_linear = signal_power_raw / (noise_power_raw + 1e-12)
    snr_raw_db = 10.0 * np.log10(snr_raw_linear + 1e-12)
    
    # Compute post-equalization SNR (rough estimate)
    signal_power_eq = np.mean(np.abs(x_eq)**2)
    # Noise power after EQ ≈ noise_variance * mean(|eq_filter|²)
    eq_filter_power = np.mean(np.abs(eq_filter)**2)
    noise_power_eq = noise_variance * eq_filter_power
    snr_eq_linear = signal_power_eq / (noise_power_eq + 1e-12)
    snr_eq_db = 10.0 * np.log10(snr_eq_linear + 1e-12)
    
    # 🔧 FINAL CHECK: Compute SNR improvement (for logging)
    snr_improvement_db = snr_eq_db - snr_raw_db
    
    # Blending for low SNR (prevent noise amplification)
    # 🔧 IMPROVED: More aggressive equalization for better pattern preservation
    if blend_factor is None:
        # Auto-compute blend factor based on SNR
        # Increased blend factors for better pattern recovery
        if snr_db < 3.0:
            blend_factor = 0.7  # 70% equalized, 30% raw (increased from 0.5)
        elif snr_db < 5.0:
            blend_factor = 0.85  # 85% equalized (increased from 0.7)
        else:
            blend_factor = 1.0  # Full equalization
    else:
        blend_factor = blend_factor
    
    # Apply blending if needed
    if blend_factor < 1.0:
        x_eq = blend_factor * x_eq + (1.0 - blend_factor) * rx_grid
    
    eq_info = {
        'alpha_used': float(alpha_reg),
        'snr_input_db': float(snr_db),
        'snr_raw_db': float(snr_raw_db),  # 🔧 FINAL CHECK: Pre-EQ SNR
        'snr_eq_db': float(snr_eq_db),
        'snr_improvement_db': float(snr_improvement_db),  # 🔧 FINAL CHECK: SNR gain
        'blend_factor': float(blend_factor),
        'eq_filter_power': float(eq_filter_power),
        'alpha_is_per_subcarrier': False,  # 🔧 FINAL CHECK: Currently global α
        'alpha_units': 'power'  # 🔧 FINAL CHECK: α in power units (noise_variance)
    }
    
    return x_eq, eq_info


def compute_pattern_preservation(tx_grid, rx_raw, rx_eq, target_subcarriers=None):
    """
    Compute pattern preservation metrics after equalization.
    
    Args:
        tx_grid: Transmitted grid (with injection pattern)
        rx_raw: Raw received grid (before equalization)
        rx_eq: Equalized received grid
        target_subcarriers: List of subcarrier indices where pattern should be (default: 24-39)
    
    Returns:
        dict: Preservation metrics (correlation, energy_ratio, etc.)
    """
    if target_subcarriers is None:
        target_subcarriers = np.arange(24, 40)
    
    # Extract pattern from TX (difference between attack and benign would be ideal,
    # but we use magnitude as proxy)
    tx_pattern = np.abs(tx_grid)
    
    # Compute correlation and energy in target subcarriers
    metrics = {}
    
    # Energy in target subcarriers
    tx_energy_target = np.mean(np.abs(tx_pattern[:, target_subcarriers])**2)
    rx_raw_energy_target = np.mean(np.abs(rx_raw[:, target_subcarriers])**2)
    rx_eq_energy_target = np.mean(np.abs(rx_eq[:, target_subcarriers])**2)
    
    # Energy ratio (should be ≥ 1.0 for good preservation)
    metrics['energy_ratio_raw'] = float(rx_raw_energy_target / (tx_energy_target + 1e-12))
    metrics['energy_ratio_eq'] = float(rx_eq_energy_target / (tx_energy_target + 1e-12))
    
    # Correlation in target subcarriers (flatten for correlation)
    tx_flat = tx_pattern[:, target_subcarriers].flatten()
    rx_raw_flat = np.abs(rx_raw[:, target_subcarriers]).flatten()
    rx_eq_flat = np.abs(rx_eq[:, target_subcarriers]).flatten()
    
    # Normalize for correlation
    tx_norm = (tx_flat - np.mean(tx_flat)) / (np.std(tx_flat) + 1e-12)
    rx_raw_norm = (rx_raw_flat - np.mean(rx_raw_flat)) / (np.std(rx_raw_flat) + 1e-12)
    rx_eq_norm = (rx_eq_flat - np.mean(rx_eq_flat)) / (np.std(rx_eq_flat) + 1e-12)
    
    # Correlation
    corr_raw = np.mean(tx_norm * rx_raw_norm)
    corr_eq = np.mean(tx_norm * rx_eq_norm)
    
    metrics['correlation_raw'] = float(corr_raw)
    metrics['correlation_eq'] = float(corr_eq)
    metrics['correlation_improvement'] = float(corr_eq - corr_raw)
    
    # Overall preservation score (0-1, higher is better)
    # Combine correlation and energy ratio
    preservation_raw = max(0.0, min(1.0, (corr_raw + 1.0) / 2.0))  # Normalize [-1,1] to [0,1]
    preservation_eq = max(0.0, min(1.0, (corr_eq + 1.0) / 2.0))
    
    metrics['preservation_raw'] = float(preservation_raw)
    metrics['preservation_eq'] = float(preservation_eq)
    metrics['preservation_improvement'] = float(preservation_eq - preservation_raw)
    
    return metrics

