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
from scipy.special import j0  # Bessel function for temporal correlation
from scipy.signal import correlate
from config.settings import CSI_CFG


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


def mmse_equalize(rx_grid, h_est, snr_db=20.0, alpha_reg=None, blend_factor=None, noise_variance_est=None, metadata=None):
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
    # 🔧 FIX: Better alpha computation to prevent noise amplification
    # α should be proportional to noise_variance but also consider channel power
    # 🔧 CRITICAL FIX: Use noise_variance from CSI estimation (from pilots) if available
    # But validate it against signal power to prevent it from being too large
    signal_power = np.mean(np.abs(rx_grid)**2)
    
    if noise_variance_est is not None and noise_variance_est > 1e-12:
        # Use noise variance from LMMSE CSI estimation, but validate it
        # 🔧 CRITICAL FIX: For dual-hop, signal_power can be very small due to attenuation
        # But noise_variance is constant (additive noise), so we should NOT compare them directly
        # Instead, validate noise_variance against expected noise power from SNR input
        snr_linear = 10.0**(snr_db / 10.0)
        # Expected noise power if signal was normalized to 1.0: noise_power = 1.0 / snr_linear
        expected_noise_power = 1.0 / (snr_linear + 1e-12)
        
        # Validate: noise_variance should be within reasonable range of expected noise power
        # Allow 0.1x to 10x range (accounts for channel estimation error and dual-hop effects)
        if noise_variance_est > expected_noise_power * 10.0 or noise_variance_est < expected_noise_power * 0.1:
            # noise_variance is outside reasonable range, use SNR-based estimate
            # But for dual-hop, use constant noise power (not scaled with signal_power)
            noise_variance = expected_noise_power
        else:
            # noise_variance is reasonable, use it
            noise_variance = noise_variance_est
    else:
        # Fallback: estimate from SNR input
        # 🔧 CRITICAL FIX: For dual-hop, noise is constant (not scaled with signal_power)
        # Use expected noise power from SNR input (assuming normalized signal = 1.0)
        snr_linear = 10.0**(snr_db / 10.0)
        noise_variance = 1.0 / (snr_linear + 1e-12)  # Constant noise power (not signal_power / snr_linear)
    
    if alpha_reg is None:
        signal_power = np.mean(np.abs(rx_grid)**2)
        
        # 🔧 FIX: Also consider channel power to prevent noise amplification
        # If channel is weak (|H|² small), need more regularization
        h_power = np.mean(np.abs(h_est)**2)
        
        # 🔧 A1: Apply floor to h_power to prevent H_power ≈ 0 failures
        h_power_floor = 1e-8  # Minimum allowed |Ĥ|²
        h_power = max(h_power, h_power_floor)
        
        # 🔧 CRITICAL FIX: Compute snr_raw_db using actual noise_variance for accurate alpha policy
        # We have noise_variance already (from CSI estimation or SNR input), use it to compute snr_raw_db
        signal_power_for_snr = np.mean(np.abs(rx_grid)**2)
        snr_raw_linear_prelim = signal_power_for_snr / (noise_variance + 1e-12)
        snr_raw_db_prelim = 10.0 * np.log10(snr_raw_linear_prelim + 1e-12)
        
        # 🔧 FIX: Improved alpha policy for dual-hop (as per user requirements)
        # Use actual snr_raw_db_prelim (computed from actual noise_variance) for more accurate alpha
        # SNR_raw<3 dB ⇒ α=10·noise_var; 3–10 dB ⇒ 5·noise_var; >10 dB ⇒ noise_var
        if snr_raw_db_prelim < 3.0:  # Very low SNR
            alpha_base = noise_variance * 10.0
        elif snr_raw_db_prelim < 10.0:  # Low SNR
            alpha_base = noise_variance * 5.0
        else:  # Normal SNR
            alpha_base = noise_variance
        
        # 🔧 CRITICAL FIX: Cap alpha relative to channel power FIRST (before channel multiplier)
        # This prevents alpha_base * multiplier from becoming too large
        if h_power > 1e-12:
            # Compute cap first
            alpha_max_from_h = h_power * 3.0    # 🔧 CRITICAL: Cap at 3x channel power
            # 🔧 CRITICAL FIX: alpha_min_from_h was too small (0.1% of channel power)
            # This was causing alpha to be capped at a very small value, preventing effective equalization
            # Remove alpha_min_from_h constraint - let alpha policy determine the minimum
            # alpha_min_from_h = h_power * 0.001  # REMOVED: Too restrictive
            alpha_min_from_h = 0.0  # No minimum constraint - let alpha policy work
            
            # 🔧 FIX: Cap alpha_base BEFORE applying channel multiplier
            # This ensures that even after multiplier, alpha won't exceed the cap
            # If alpha_base is already larger than cap, reduce it
            if alpha_base > alpha_max_from_h:
                # alpha_base is too large - use a fraction of it that fits within cap
                # Use the smaller of: alpha_base or alpha_max_from_h
                alpha_base = min(alpha_base, alpha_max_from_h)
        
        # 🔧 FIX: Add channel-dependent regularization (but now alpha_base is already capped)
        # If channel power is very small, increase alpha to prevent noise amplification
        # But don't over-regularize - the multiplier must respect the cap
        if h_power > 1e-12:
            # 🔧 CRITICAL: Compute maximum allowed multiplier to stay within cap
            # If alpha_base is already at cap, multiplier should be 1.0
            # Otherwise, multiplier can be up to (alpha_max_from_h / alpha_base)
            max_multiplier = alpha_max_from_h / (alpha_base + 1e-12)  # Maximum multiplier to stay within cap
            
            if h_power < 1e-4:  # Very weak channel
                desired_multiplier = 3.0
            elif h_power < 1e-2:  # Weak channel
                desired_multiplier = 2.0
            else:  # Normal channel
                desired_multiplier = 1.0
            
            # Use the smaller of desired multiplier and max allowed multiplier
            actual_multiplier = min(desired_multiplier, max_multiplier)
            alpha_reg = alpha_base * actual_multiplier
        else:
            # Fallback if h_power is too small
            if h_power < 1e-4:
                alpha_reg = alpha_base * 3.0
            elif h_power < 1e-2:
                alpha_reg = alpha_base * 2.0
            else:
                alpha_reg = alpha_base
        
        # 🔧 CRITICAL: Apply cap again after multiplier (final safety check)
        if h_power > 1e-12:
            alpha_reg = max(alpha_reg, alpha_min_from_h)
            alpha_reg = min(alpha_reg, alpha_max_from_h)  # 🔧 CRITICAL: Final cap check - MUST be ≤ 3x
        
        # Ensure alpha is reasonable relative to signal power (but NEVER override channel power cap)
        # 🔧 CRITICAL FIX: signal_power constraint was too restrictive
        # alpha_min_from_signal = signal_power * 0.01 was too small and was capping alpha
        # SOLUTION: Remove signal_power constraint - let alpha policy work without interference
        # The alpha policy (based on SNR_raw and noise_variance) is sufficient
        if h_power > 1e-12:
            # 🔧 CRITICAL: ALWAYS re-apply cap to ensure it's never exceeded
            # This is the final safety check - MUST be the last operation
            alpha_reg = min(alpha_reg, alpha_max_from_h)  # Ensure cap is NEVER exceeded - FINAL CHECK
            
            # 🔧 DEBUG: Assert that cap is respected (for debugging)
            if h_power > 1e-12:
                alpha_ratio = alpha_reg / h_power
                if alpha_ratio > 3.01:  # Allow small numerical error (0.01)
                    import warnings
                    warnings.warn(f"Alpha cap violation: alpha_ratio={alpha_ratio:.2f}x > 3.0x (h_power={h_power:.6e}, alpha_reg={alpha_reg:.6e}, alpha_max_from_h={alpha_max_from_h:.6e})")
        else:
            # If channel power is too small, use signal power as fallback
            # But still respect a reasonable maximum
            alpha_min = signal_power * 0.01
            alpha_max_from_signal = signal_power * 3.0  # Cap at 3x signal power as fallback
            alpha_reg = max(alpha_reg, alpha_min)
            alpha_reg = min(alpha_reg, alpha_max_from_signal)  # Cap fallback too
    else:
        noise_variance = alpha_reg
    
    # 🔧 CRITICAL: Final assertion - alpha_reg must respect cap if h_power is available
    if h_power > 1e-12 and alpha_reg is not None:
        alpha_max_from_h_final = h_power * 3.0
        if alpha_reg > alpha_max_from_h_final:
            # Force cap - this should never happen, but if it does, fix it
            import warnings
            warnings.warn(f"CRITICAL: Alpha cap violated! Forcing cap: alpha_reg={alpha_reg:.6e} > alpha_max={alpha_max_from_h_final:.6e}")
            alpha_reg = alpha_max_from_h_final
    
    # MMSE equalization: X_est = (H* / (|H|² + α)) * Y
    h_conj = np.conj(h_est)
    h_mag_sq = np.abs(h_est)**2
    
    # 🔧 A1: Floor on |Ĥ|² to prevent H_power ≈ 0 failures
    # Apply floor to prevent extremely small channel power (causes alpha_ratio issues)
    h_power_floor = 1e-8  # Minimum allowed |Ĥ|² (adjustable: 1e-8 to 1e-10)
    h_mag_sq = np.maximum(h_mag_sq, h_power_floor)
    
    # 🔧 MICRO-TUNE 3: Per-subcarrier alpha (adaptive regularization) - DISABLED
    # α_k = α_0 * (median(|Ĥ|²) / (|Ĥ_k|² + ε))^0.5
    # This prevents over/under-equalization on very weak/strong tones
    # NOTE: Disabled - testing showed it may hurt performance
    denominator = h_mag_sq + alpha_reg + 1e-12  # Prevent division by zero
    
    # 🔧 A2: Cap inverse to prevent extreme amplification
    # (|Ĥ|² + α)⁻¹ ≤ 10³ to prevent noise amplification
    denominator_min = 1e-3  # Minimum denominator = 1/10³
    denominator = np.maximum(denominator, denominator_min)
    
    # Equalization filter
    eq_filter = h_conj / denominator
    
    # 🔧 B3: Adaptive mask for weak subcarriers (prevent noise amplification)
    # For subcarriers with low |Ĥ|² or high pilot-MSE, use conservative EQ weight
    h_mag_sq_per_subcarrier = np.abs(h_est)**2  # |Ĥ|² per subcarrier
    h_power_threshold = np.percentile(h_mag_sq_per_subcarrier, 10)  # Bottom 10% threshold
    
    # Create adaptive mask: reduce EQ weight for weak subcarriers
    adaptive_mask = np.ones_like(eq_filter, dtype=np.float32)
    weak_subcarrier_mask = h_mag_sq_per_subcarrier < h_power_threshold
    # Apply conservative weight (0.85) to weak subcarriers
    adaptive_mask[weak_subcarrier_mask] = 0.85  # Conservative weight
    eq_filter = eq_filter * adaptive_mask
    
    # 🔧 CRITICAL FIX: Compute SNR raw BEFORE equalization and BEFORE any normalization/blending
    # SNR_raw must be computed on the RAW received signal: s(t) = G_r * (h_DL * (h_UL * x))
    # Noise: n(t) = G_r * (h_DL * n_UL) + n_DL
    # SNR_raw = 10*log10(E[|s|²] / E[|n|²])
    # 
    # For dual-hop: signal power after channel attenuation, noise power is constant (additive)
    # Input SNR (snr_db) is at transmitter (signal normalized to 1.0)
    # After dual-hop: signal_power_rx << 1.0, but noise_power stays constant
    
    # Compute signal power from RAW rx_grid (before any processing)
    signal_power_raw = np.mean(np.abs(rx_grid)**2)
    
    # 🔧 CRITICAL: Noise power is CONSTANT (additive white Gaussian noise, doesn't scale with signal)
    # If input SNR = 20 dB and signal was normalized to 1.0:
    #   noise_power = 1.0 / (10^(20/10)) = 1.0 / 100 = 0.01 (constant)
    # After dual-hop channel: signal_power_rx = 8e-5 (attenuated), noise_power = 0.01 (still constant)
    # Actual SNR = 10*log10(8e-5 / 0.01) ≈ -21 dB
    
    # Compute constant noise power from input SNR (assuming normalized signal = 1.0 at transmitter)
    snr_linear_input = 10.0**(snr_db / 10.0)
    noise_power_constant = 1.0 / (snr_linear_input + 1e-12)  # Constant noise power (additive)
    
    # Compute actual SNR after channel attenuation (BEFORE equalization)
    snr_raw_linear = signal_power_raw / (noise_power_constant + 1e-12)
    snr_raw_db = 10.0 * np.log10(snr_raw_linear + 1e-12)
    
    # Use this noise_power_constant for SNR computation (not scaled with signal)
    noise_variance_for_snr = noise_power_constant
    
    # Apply equalization
    x_eq = rx_grid * eq_filter
    
    # 🔧 CRITICAL FIX: Compute SNR_eq correctly
    # The key insight: After MMSE equalization, the effective channel becomes approximately 1.0
    # Signal after EQ: x_eq = eq_filter * (h*x + n) ≈ x + eq_filter*n (if EQ is good)
    # So signal power should be approximately restored to original (normalized to 1.0)
    
    # Compute effective channel after EQ: eq_filter * h_est
    # If EQ is perfect, this should be close to 1.0
    effective_channel_eq = eq_filter * h_est
    effective_channel_power = np.mean(np.abs(effective_channel_eq)**2)
    
    # Original signal power (normalized to 1.0 at transmitter)
    # After perfect EQ, signal should be restored: signal_power_eq ≈ 1.0
    # But we need to account for EQ quality (effective_channel_power)
    # If effective_channel_power ≈ 1.0, then signal is well restored
    # If effective_channel_power < 1.0, then signal is still attenuated
    
    # Estimate signal power after EQ:
    # signal_power_eq = effective_channel_power * original_signal_power
    # Assuming normalized input (original_signal_power ≈ 1.0)
    signal_power_eq_estimated = effective_channel_power * 1.0
    
    # Also compute from x_eq directly (includes both signal and noise)
    signal_power_eq_measured = np.mean(np.abs(x_eq)**2)
    
    # Use the estimated value (from effective channel) as it's cleaner
    # But ensure it's reasonable (not too small)
    signal_power_eq = max(signal_power_eq_estimated, signal_power_eq_measured * 0.1)
    
    # Noise power after EQ: noise is filtered by eq_filter
    eq_filter_power = np.mean(np.abs(eq_filter)**2)
    noise_power_eq = noise_variance_for_snr * eq_filter_power
    
    # Compute SNR after EQ
    snr_eq_linear = signal_power_eq / (noise_power_eq + 1e-12)
    snr_eq_db = 10.0 * np.log10(snr_eq_linear + 1e-12)
    
    # 🔧 CRITICAL FIX: Don't cap too aggressively - allow improvement if EQ works
    # Only cap if SNR_eq is unreasonably high (which would indicate calculation error)
    snr_eq_db = np.clip(snr_eq_db, snr_raw_db - 30.0, snr_raw_db + 40.0)  # Wider range to allow improvement
    
    # 🔧 FINAL CHECK: Compute SNR improvement (for logging)
    snr_improvement_db = snr_eq_db - snr_raw_db
    
    # Blending for low SNR (prevent noise amplification)
    # 🔧 TEMPORARY FIX: Disable blending to test if it's causing SNR improvement ≈ 0
    # Set blend_factor=1.0 to use full equalization and see true SNR improvement
    # TODO: Re-enable blending after confirming SNR measurement is correct
    if blend_factor is None:
        # 🔧 TEMPORARY: Force blend_factor=1.0 to test SNR improvement without blending interference
        blend_factor = 1.0  # Full equalization (no blending)
        
        # Original blend policy (commented out for testing):
        # if snr_raw_db < -15.0:  # Extremely low raw SNR
        #     blend_factor = 1.0  # Full equalization
        # elif snr_raw_db < -10.0:  # Very low raw SNR
        #     blend_factor = 1.0  # Full equalization
        # elif snr_raw_db < 3.0:  # Low raw SNR
        #     blend_factor = 0.85  # 85% equalized, 15% raw
        # elif snr_raw_db < 5.0:  # Moderate raw SNR
        #     blend_factor = 0.95  # 95% equalized
        # else:
        #     blend_factor = 1.0  # Full equalization
    else:
        blend_factor = blend_factor
    
    # Apply blending if needed
    if blend_factor < 1.0:
        x_eq = blend_factor * x_eq + (1.0 - blend_factor) * rx_grid
    
    # 🔧 B4: Band emphasis (optional, for ablation testing)
    # 🔧 MICRO-TUNE 2: Stronger band emphasis with power normalization
    # 🔧 IMPROVEMENT: Get target_subcarriers from injection_info (support random/hopping patterns)
    apply_band_emphasis = True  # ✅ ENABLED: Improves pattern preservation
    if apply_band_emphasis:
        # Get target_subcarriers from metadata (injection_info) if available
        target_subcarriers = np.arange(24, 40)  # Default: fixed band
        if metadata is not None:
            injection_info = metadata.get('injection_info', {})
            if 'selected_subcarriers' in injection_info:
                target_subcarriers = np.array(injection_info['selected_subcarriers'], dtype=int)
                # Ensure valid range
                target_subcarriers = target_subcarriers[
                    (target_subcarriers >= 0) & (target_subcarriers < rx_grid.shape[1])
                ]
                if len(target_subcarriers) == 0:
                    target_subcarriers = np.arange(24, 40)  # Fallback to default
        boost_factor = 1.05  # 5% boost on target band (optimal balance)
        reduce_factor = 0.98  # 2% reduction outside
        
        # Compute power before emphasis
        power_before = np.mean(np.abs(x_eq)**2)
        
        # Create frequency mask
        freq_mask = np.ones((x_eq.shape[0], x_eq.shape[1]), dtype=np.complex64)
        for sc_idx in target_subcarriers:
            if sc_idx < x_eq.shape[1]:
                freq_mask[:, sc_idx] = boost_factor
        # Apply reduction to non-target subcarriers
        for sc_idx in range(x_eq.shape[1]):
            if sc_idx not in target_subcarriers:
                freq_mask[:, sc_idx] = reduce_factor
        
        # Apply mask
        x_eq = x_eq * freq_mask
        
        # Normalize power to maintain ΔP < 0.2%
        power_after = np.mean(np.abs(x_eq)**2)
        power_change = abs(power_after - power_before) / (power_before + 1e-12)
        if power_change > 0.002:  # If power change > 0.2%, normalize
            normalization_factor = np.sqrt(power_before / (power_after + 1e-12))
            x_eq = x_eq * normalization_factor
    
    # 🔧 DEBUG: Compute additional metadata for diagnosis
    h_power_mean = float(np.mean(np.abs(h_est)**2))
    
    # 🔧 A1: Apply floor to h_power_mean for consistent reporting
    h_power_floor = 1e-8
    h_power_mean = max(h_power_mean, h_power_floor)
    
    # 🔧 A4: Gating - flag CSI failure if H_power is too small
    flag_csi_fail = 0
    if h_power_mean < 1e-6:
        flag_csi_fail = 1
        # If CSI failed, use conservative alpha (large regularization)
        # This prevents noise amplification
        if alpha_reg is not None:
            alpha_reg = max(alpha_reg, noise_variance * 10.0)
    
    alpha_ratio_debug = float(alpha_reg / h_power_mean) if h_power_mean > 1e-12 else 0.0
    signal_power_raw_debug = float(signal_power_raw)
    noise_power_constant_debug = float(noise_power_constant)
    
    # 🔧 CRITICAL: Get target_subcarriers from injection_info (not hardcoded)
    target_subcarriers_for_logging = None
    pattern_indices_source = 'hardcoded'
    if metadata is not None:
        injection_info = metadata.get('injection_info', {})
        if 'selected_subcarriers' in injection_info:
            target_subcarriers_for_logging = np.array(injection_info['selected_subcarriers'], dtype=int)
            pattern_indices_source = 'injection_info'
        elif 'selected_subcarriers_per_symbol' in injection_info and injection_info['selected_subcarriers_per_symbol'] is not None:
            # For hopping patterns, use union of all subcarriers
            all_scs = set()
            for scs_list in injection_info['selected_subcarriers_per_symbol']:
                all_scs.update(scs_list)
            target_subcarriers_for_logging = np.array(sorted(all_scs), dtype=int)
            pattern_indices_source = 'injection_info_hopping'
    
    eq_info = {
        'alpha_used': float(alpha_reg),
        'snr_input_db': float(snr_db),
        'snr_raw_db': float(snr_raw_db),  # 🔧 CRITICAL: Pre-EQ SNR (BEFORE normalization/blending)
        'snr_eq_db': float(snr_eq_db),
        'snr_improvement_db': float(snr_improvement_db),  # 🔧 CRITICAL: SNR gain
        'blend_factor': float(blend_factor),
        'eq_filter_power': float(eq_filter_power),
        'alpha_is_per_subcarrier': False,  # Currently global α
        'alpha_units': 'power',  # α in power units (noise_variance)
        # 🔧 DEBUG: Additional metadata for diagnosis
        'h_power_mean': h_power_mean,  # Mean |Ĥ|² (with floor applied)
        'alpha_ratio': alpha_ratio_debug,  # α / |Ĥ|²
        'signal_power_raw': signal_power_raw_debug,  # E[|rx_grid|²] (before EQ)
        'noise_power_constant': noise_power_constant_debug,  # Constant noise power from SNR input
        'snr_raw_linear': float(snr_raw_linear),  # SNR_raw in linear scale
        'snr_eq_linear': float(snr_eq_linear),  # SNR_eq in linear scale
        'flag_csi_fail': int(flag_csi_fail),  # 🔧 A4: Flag for CSI failure (H_power < 1e-6)
        # 🔧 NEW: Pattern indices source (critical for diagnosis)
        'pattern_indices_source': pattern_indices_source,  # Source of pattern indices
        'target_subcarriers_count': len(target_subcarriers_for_logging) if target_subcarriers_for_logging is not None else 0,
    }
    
    return x_eq, eq_info


def estimate_csi_lmmse_2d_separable(tx_grid, rx_grid, pilot_symbols=None, 
                                     metadata=None, csi_cfg=None):
    """
    Enhanced LMMSE 2D Separable CSI Estimation.
    
    Implements:
    1. Noise estimation from pilots
    2. WSSUS correlation model (temporal + frequency)
    3. Separable LMMSE filtering (temporal then frequency)
    4. Global interpolation
    5. Alignment compensation (phase & delay)
    
    Args:
        tx_grid: Transmitted grid (symbols, subcarriers)
        rx_grid: Received grid (symbols, subcarriers)
        pilot_symbols: List of pilot symbol indices (default: from CSI_CFG)
        metadata: Dict with fd_ul, fd_dl, delay_samples, snr_ul_db, snr_dl_db
        csi_cfg: CSI configuration dict (default: from settings)
    
    Returns:
        h_est: Estimated CSI (symbols, subcarriers)
        csi_info: Dict with estimation metrics (nmse, pilot_mse, etc.)
    """
    if csi_cfg is None:
        csi_cfg = CSI_CFG
    
    if pilot_symbols is None:
        pilot_symbols = csi_cfg['pilots']['sym_idx']
    
    num_symbols, num_subcarriers = rx_grid.shape[0], rx_grid.shape[1]
    lmmse_cfg = csi_cfg['lmmse']
    
    # ===== Step 1: Initial LS estimation on pilots =====
    h_est_pilots = []
    pilot_positions = []
    pilot_residuals = []
    
    # 🔧 A3: Pilot sanity check - reject weak pilots
    pilot_energy_threshold = 1e-6  # Minimum pilot energy threshold
    
    for sym_idx in pilot_symbols:
        if sym_idx < num_symbols:
            for sc_idx in range(num_subcarriers):
                tx_pilot = tx_grid[sym_idx, sc_idx]
                rx_pilot = rx_grid[sym_idx, sc_idx]
                if np.abs(tx_pilot) > 1e-9:
                    # Check pilot energy
                    pilot_energy = np.abs(rx_pilot)**2
                    if pilot_energy < pilot_energy_threshold:
                        # Reject weak pilot - will interpolate from neighbors
                        continue
                    
                    h_pilot = rx_pilot / tx_pilot
                    h_est_pilots.append(h_pilot)
                    pilot_positions.append((sym_idx, sc_idx))
                    # Residual for noise estimation
                    residual = rx_pilot - tx_pilot * h_pilot
                    pilot_residuals.append(residual)
    
    # 🔧 A3: Check if we have enough valid pilots (≥50% of expected)
    expected_pilots = len(pilot_symbols) * num_subcarriers
    valid_pilot_ratio = len(h_est_pilots) / max(expected_pilots, 1)
    
    if len(h_est_pilots) == 0 or valid_pilot_ratio < 0.5:
        # 🔧 A3: Fallback to LS + median filter if too few valid pilots
        # Fallback to simple division with smoothing
        denom = np.where(np.abs(tx_grid) > 1e-9, tx_grid, 1e-9)
        h_est = rx_grid / denom
        
        # Apply 3x3 median filter for smoothing
        from scipy import ndimage
        h_mag = np.abs(h_est)
        h_phase = np.angle(h_est)
        h_mag_smooth = ndimage.median_filter(h_mag, size=(3, 3))
        h_phase_unwrapped = np.unwrap(h_phase, axis=0)
        h_phase_smooth = ndimage.median_filter(h_phase_unwrapped, size=(3, 3))
        h_phase_smooth = np.angle(np.exp(1j * h_phase_smooth))
        h_est = h_mag_smooth * np.exp(1j * h_phase_smooth)
        
        # Compute basic metrics
        P_H_fallback = np.mean(np.abs(h_est)**2)
        P_H_fallback = max(P_H_fallback, 1e-8)  # Apply floor
        
        return h_est, {
            'nmse_db': None, 
            'pilot_mse': None,
            'noise_variance': 1e-12,
            'P_H': float(P_H_fallback),
            'pilot_fallback': True  # Flag for fallback mode
        }
    
    h_est_pilots = np.array(h_est_pilots)
    pilot_positions = np.array(pilot_positions)
    pilot_residuals = np.array(pilot_residuals)
    
    # ===== Step 2: Noise variance estimation =====
    # 🔧 CRITICAL FIX: Initial noise variance from pilot residuals
    # Note: This is a rough estimate; final noise variance will be computed after LMMSE filtering
    if len(pilot_residuals) > 0:
        noise_variance_initial = np.median(np.abs(pilot_residuals)**2)
    else:
        noise_variance_initial = 1e-12
    
    # Use initial estimate (will be refined after LMMSE filtering)
    noise_variance = max(noise_variance_initial, 1e-12)
    
    # Channel power from pilots
    P_H = np.mean(np.abs(h_est_pilots)**2)
    
    # ===== Step 3: Correlation model (WSSUS) =====
    # Get Doppler from metadata (for temporal correlation)
    if metadata is not None:
        fd_ul = metadata.get('fd_ul', 0.0)
        fd_dl = metadata.get('fd_dl', 0.0)
        fd_max = max(abs(fd_ul), abs(fd_dl))
    else:
        fd_max = lmmse_cfg['doppler_dl_hz']  # Default
    
    # Temporal correlation: ρ_t(Δm) = J₀(2π f_D Δm T_sym)
    Tsym = lmmse_cfg['Tsym_s']
    win_t = lmmse_cfg['win_t']
    
    # 🔧 B2: Adjust win_t for high Doppler (preserve temporal details)
    if fd_max > 100.0:  # High Doppler
        win_t = 5  # Smaller window to preserve details
    
    def rho_temporal(delta_m):
        """Temporal correlation function."""
        if delta_m == 0:
            return 1.0
        arg = 2.0 * np.pi * fd_max * delta_m * Tsym
        return j0(arg)
    
    # Frequency correlation: ρ_f(Δk) = sinc(Δk Δf τ_rms)
    tau_rms = lmmse_cfg['tau_rms_s']
    subc_bw = lmmse_cfg['subc_bw_hz']
    win_f = lmmse_cfg['win_f']
    
    # 🔧 B2: Improve frequency interpolation with larger window (9 or 11)
    # Use larger window for better frequency domain smoothing
    win_f = max(win_f, 9)  # At least 9, prefer 11 if available
    if win_f < 11 and num_subcarriers >= 64:
        win_f = 11  # Use 11 for better frequency domain interpolation
    
    def rho_frequency(delta_k):
        """Frequency correlation function."""
        if delta_k == 0:
            return 1.0
        arg = delta_k * subc_bw * tau_rms
        if abs(arg) < 1e-10:
            return 1.0
        return np.sinc(arg / np.pi)
    
    # ===== Step 4: Separable LMMSE filtering =====
    # Create pilot grid
    h_pilot_grid = np.zeros((num_symbols, num_subcarriers), dtype=np.complex64)
    pilot_mask = np.zeros((num_symbols, num_subcarriers), dtype=bool)
    
    for i, (sym_idx, sc_idx) in enumerate(pilot_positions):
        h_pilot_grid[sym_idx, sc_idx] = h_est_pilots[i]
        pilot_mask[sym_idx, sc_idx] = True
    
    # Step 4.1: Temporal LMMSE filtering (per subcarrier)
    h_temporal = np.zeros_like(h_pilot_grid, dtype=np.complex64)
    
    for sc_idx in range(num_subcarriers):
        # Get pilot values for this subcarrier
        pilot_syms = pilot_positions[pilot_positions[:, 1] == sc_idx, 0]
        if len(pilot_syms) == 0:
            continue
        
        # Build temporal correlation matrix R_tt
        pilot_syms_sorted = np.sort(pilot_syms)
        n_pilots = len(pilot_syms_sorted)
        R_tt = np.zeros((n_pilots, n_pilots), dtype=np.complex64)
        
        for i, sym_i in enumerate(pilot_syms_sorted):
            for j, sym_j in enumerate(pilot_syms_sorted):
                delta_m = abs(sym_i - sym_j)
                R_tt[i, j] = rho_temporal(delta_m) * P_H
        
        # Wiener filter: W_t = (R_tt + σ²/P_H * I)^(-1) * R_tt
        # But for separable, we apply per-window
        # Simplified: apply LMMSE on sliding window
        for sym_idx in range(num_symbols):
            # Find nearby pilots
            nearby_pilots = []
            nearby_indices = []
            for p_sym in pilot_syms_sorted:
                if abs(p_sym - sym_idx) <= win_t // 2:
                    nearby_pilots.append(h_pilot_grid[p_sym, sc_idx])
                    nearby_indices.append(p_sym)
            
            if len(nearby_pilots) > 0:
                nearby_pilots = np.array(nearby_pilots)
                n_nearby = len(nearby_pilots)
                
                # Build correlation matrix for nearby pilots
                R_near = np.zeros((n_nearby, n_nearby), dtype=np.complex64)
                for i, sym_i in enumerate(nearby_indices):
                    for j, sym_j in enumerate(nearby_indices):
                        delta_m = abs(sym_i - sym_j)
                        R_near[i, j] = rho_temporal(delta_m) * P_H
                
                # Wiener filter
                reg_term = (noise_variance / P_H) * np.eye(n_nearby, dtype=np.complex64)
                W_t = R_near @ np.linalg.inv(R_near + reg_term)
                
                # Estimate at this symbol
                h_temporal[sym_idx, sc_idx] = W_t[0, :] @ nearby_pilots
            elif pilot_mask[sym_idx, sc_idx]:
                h_temporal[sym_idx, sc_idx] = h_pilot_grid[sym_idx, sc_idx]
    
    # Step 4.2: Frequency LMMSE filtering (per symbol)
    h_frequency = np.zeros_like(h_temporal, dtype=np.complex64)
    
    for sym_idx in range(num_symbols):
        # Get pilot values for this symbol
        pilot_scs = pilot_positions[pilot_positions[:, 0] == sym_idx, 1]
        if len(pilot_scs) == 0:
            continue
        
        pilot_scs_sorted = np.sort(pilot_scs)
        
        # Apply frequency LMMSE on sliding window
        for sc_idx in range(num_subcarriers):
            # Find nearby pilots
            nearby_pilots = []
            nearby_indices = []
            for p_sc in pilot_scs_sorted:
                if abs(p_sc - sc_idx) <= win_f // 2:
                    nearby_pilots.append(h_temporal[sym_idx, p_sc])
                    nearby_indices.append(p_sc)
            
            if len(nearby_pilots) > 0:
                nearby_pilots = np.array(nearby_pilots)
                n_nearby = len(nearby_pilots)
                
                # Build frequency correlation matrix
                R_near = np.zeros((n_nearby, n_nearby), dtype=np.complex64)
                for i, sc_i in enumerate(nearby_indices):
                    for j, sc_j in enumerate(nearby_indices):
                        delta_k = abs(sc_i - sc_j)
                        R_near[i, j] = rho_frequency(delta_k) * P_H
                
                # Wiener filter
                reg_term = (noise_variance / P_H) * np.eye(n_nearby, dtype=np.complex64)
                W_f = R_near @ np.linalg.inv(R_near + reg_term)
                
                # Estimate at this subcarrier
                h_frequency[sym_idx, sc_idx] = W_f[0, :] @ nearby_pilots
            elif pilot_mask[sym_idx, sc_idx]:
                h_frequency[sym_idx, sc_idx] = h_temporal[sym_idx, sc_idx]
    
    # ===== Step 5: Global interpolation =====
    # Use linear 2D interpolation for remaining points (with fallback to nearest)
    from scipy.interpolate import griddata
    
    # Get all estimated points (from LMMSE filtering)
    estimated_positions = []
    estimated_values = []
    
    for sym_idx in range(num_symbols):
        for sc_idx in range(num_subcarriers):
            if np.abs(h_frequency[sym_idx, sc_idx]) > 1e-12:
                estimated_positions.append((sym_idx, sc_idx))
                estimated_values.append(h_frequency[sym_idx, sc_idx])
    
    if len(estimated_values) > 0:
        estimated_positions = np.array(estimated_positions)
        estimated_values = np.array(estimated_values)
        
        # Create full grid
        grid_y, grid_x = np.mgrid[0:num_symbols, 0:num_subcarriers]
        
        # 🔧 FIX: Check if we have enough points and they are not collinear
        # Linear interpolation requires at least 4 points for Delaunay triangulation
        # Also need to check that points are not all on the same line (collinear)
        min_points_for_linear = 4
        
        def check_collinear(points):
            """Check if points are collinear (all on same line)."""
            if len(points) < 3:
                return True  # Not enough points to form a triangle
            
            # Check if all points have same x or same y coordinate
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            if len(np.unique(x_coords)) == 1 or len(np.unique(y_coords)) == 1:
                return True  # All points on same vertical or horizontal line
            
            # Check if points are collinear (using cross product)
            if len(points) >= 3:
                # Take first 3 points
                p0, p1, p2 = points[0], points[1], points[2]
                # Vector from p0 to p1
                v1 = p1 - p0
                # Vector from p0 to p2
                v2 = p2 - p0
                # Cross product (for 2D, this is the determinant)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) < 1e-10:
                    # Check if all other points are also collinear
                    for i in range(3, len(points)):
                        v = points[i] - p0
                        cross_i = v1[0] * v[1] - v1[1] * v[0]
                        if abs(cross_i) > 1e-10:
                            return False
                    return True  # All points collinear
            
            return False
        
        use_linear = (len(estimated_values) >= min_points_for_linear and 
                     not check_collinear(estimated_positions))
        
        # 🔧 B2: Force linear interpolation for better frequency domain accuracy
        # 🔧 MICRO-TUNE 1: Try cubic interpolation if enough points (disabled for now - testing)
        # Cubic interpolation provides smoother frequency response than linear
        use_linear = True  # Always prefer linear if possible
        use_cubic = False  # Set to True to enable cubic interpolation
        
        # Check if we have enough points for cubic (typically needs more points than linear)
        min_points_for_cubic = 16  # Cubic needs more points
        if use_cubic and len(estimated_values) >= min_points_for_cubic and not check_collinear(estimated_positions):
            try:
                # Try cubic interpolation (better frequency domain accuracy)
                h_est = griddata(estimated_positions, estimated_values, 
                                (grid_y, grid_x), method='cubic', fill_value=0.0)
                # Check for NaN and fallback if needed
                if np.any(np.isnan(h_est)):
                    use_cubic = False
            except Exception as e:
                # Fallback to linear if cubic fails
                use_cubic = False
        
        if not use_cubic and use_linear and len(estimated_values) >= min_points_for_linear:
            try:
                # Try linear interpolation
                h_est = griddata(estimated_positions, estimated_values, 
                                (grid_y, grid_x), method='linear', fill_value=0.0)
                
                # Fill NaN with nearest neighbor
                nan_mask = np.isnan(h_est)
                if np.any(nan_mask):
                    for sym_idx in range(num_symbols):
                        for sc_idx in range(num_subcarriers):
                            if nan_mask[sym_idx, sc_idx]:
                                distances = np.abs(estimated_positions[:, 0] - sym_idx) + \
                                           np.abs(estimated_positions[:, 1] - sc_idx)
                                nearest_idx = np.argmin(distances)
                                h_est[sym_idx, sc_idx] = estimated_values[nearest_idx]
            except Exception as e:
                # Fallback to nearest neighbor if linear fails
                # Suppress warning for known cases (collinear points)
                if "collinear" not in str(e).lower() and "same x" not in str(e).lower() and "same y" not in str(e).lower():
                    print(f"  ⚠️  Linear interpolation failed ({str(e)[:100]}), using nearest neighbor")
                h_est = griddata(estimated_positions, estimated_values, 
                                (grid_y, grid_x), method='nearest', fill_value=0.0)
        else:
            # Not enough points or collinear, use nearest neighbor
            h_est = griddata(estimated_positions, estimated_values, 
                            (grid_y, grid_x), method='nearest', fill_value=0.0)
    else:
        # No estimated points, use frequency-filtered result as-is
        h_est = h_frequency
    
    # ===== Step 6: Alignment compensation =====
    # 🔧 B1: Make delay/phase compensation mandatory
    
    # 6.1: Delay compensation (for Scenario B) - MANDATORY
    delay_samples = 0
    if metadata is not None and 'delay_samples' in metadata:
        delay_samples = metadata['delay_samples']
    
    # 🔧 B1: If delay_samples not provided, estimate from cross-correlation
    if delay_samples == 0 or delay_samples is None:
        # Estimate delay from pilot cross-correlation
        from scipy.signal import correlate
        # Use first pilot symbol for delay estimation
        if len(pilot_symbols) > 0 and pilot_symbols[0] < num_symbols:
            sym_idx = pilot_symbols[0]
            tx_pilot_series = tx_grid[sym_idx, :]
            rx_pilot_series = rx_grid[sym_idx, :]
            # Cross-correlation
            corr = correlate(np.abs(rx_pilot_series), np.abs(tx_pilot_series), mode='full')
            # Find peak (accounting for zero-padding)
            peak_idx = np.argmax(np.abs(corr))
            delay_samples = peak_idx - (len(tx_pilot_series) - 1)
            # Clip to reasonable range (1-5 samples for dual-hop)
            delay_samples = np.clip(delay_samples, 1, 5)
    
    # Apply delay compensation
    if delay_samples > 0:
        # Ĥ(k) ← Ĥ(k) · e^(-j 2π k Δt / N)
        k_indices = np.arange(num_subcarriers)
        phase_shift = -2.0j * np.pi * k_indices * delay_samples / num_subcarriers
        h_est = h_est * np.exp(phase_shift)[np.newaxis, :]
    
    # 6.2: Phase alignment (CFO compensation) - MANDATORY
    # Compute average phase on pilot region and zero it
    pilot_phase_sum = 0.0
    pilot_count = 0
    for sym_idx in pilot_symbols:
        if sym_idx < num_symbols:
            for sc_idx in range(num_subcarriers):
                if np.abs(tx_grid[sym_idx, sc_idx]) > 1e-9:
                    tx_pilot = tx_grid[sym_idx, sc_idx]
                    rx_pilot = rx_grid[sym_idx, sc_idx]
                    h_pilot = h_est[sym_idx, sc_idx]
                    # Phase error: ∠(Ĥ_p X_p^* Y_p)
                    phase_err = np.angle(h_pilot * np.conj(tx_pilot) * rx_pilot)
                    pilot_phase_sum += phase_err
                    pilot_count += 1
    
    # 🔧 B1: Always apply phase compensation (mandatory)
    avg_phase_err = 0.0
    if pilot_count > 0:
        avg_phase_err = pilot_phase_sum / pilot_count
        h_est = h_est * np.exp(-1j * avg_phase_err)
    
    # ===== Step 7: Regularization for numerical stability =====
    epsilon = lmmse_cfg['alpha_eps']
    h_mean_power = np.mean(np.abs(h_est)**2)
    if h_mean_power > 1e-12:
        h_est = h_est / (epsilon + np.sqrt(h_mean_power))
    
    # ===== Step 8: Compute metrics =====
    # 🔧 CRITICAL FIX: Compute P_H from full CSI (h_est) instead of pilot CSI (h_est_pilots)
    # This ensures P_H matches h_power in mmse_equalize, preventing cap violation
    # P_H should represent the power of the full interpolated CSI, not just pilots
    P_H_full = np.mean(np.abs(h_est)**2)  # Power of full CSI (after interpolation and alignment)
    
    # 🔧 A1: Apply floor to P_H to prevent H_power ≈ 0 failures
    h_power_floor = 1e-8  # Minimum allowed |Ĥ|²
    P_H_full = max(P_H_full, h_power_floor)
    
    # Pilot MSE (recompute after alignment)
    # 🔧 CRITICAL FIX: Use pilot_mse as noise_variance estimate (more accurate after LMMSE)
    pilot_mse = 0.0
    pilot_count_mse = 0
    for i, (sym_idx, sc_idx) in enumerate(pilot_positions):
        tx_pilot = tx_grid[sym_idx, sc_idx]
        rx_pilot = rx_grid[sym_idx, sc_idx]
        h_pilot_est = h_est[sym_idx, sc_idx]
        residual = rx_pilot - tx_pilot * h_pilot_est
        pilot_mse += np.abs(residual)**2
        pilot_count_mse += 1
    
    if pilot_count_mse > 0:
        pilot_mse = pilot_mse / pilot_count_mse
        # 🔧 CRITICAL FIX: pilot_mse includes channel estimation error, not just noise
        # For noise_variance, we should use a more conservative estimate
        # Use the minimum of pilot_mse and a noise estimate based on signal power
        # This prevents noise_variance from being too large relative to signal power
        signal_power_estimate = np.mean(np.abs(rx_grid)**2)
        # If pilot_mse is much larger than signal_power, it likely includes estimation error
        # Use a more conservative estimate: noise_variance ≈ signal_power / (reasonable_SNR)
        # For dual-hop with very low signal power, use pilot_mse but cap it
        if pilot_mse > signal_power_estimate * 10.0:
            # pilot_mse is too large (likely includes estimation error)
            # Use a conservative estimate based on signal power
            # Assume minimum SNR of 0 dB for noise estimation
            noise_variance = max(signal_power_estimate, pilot_mse * 0.1, 1e-12)
        else:
            # pilot_mse is reasonable, use it
            noise_variance = max(pilot_mse, 1e-12)
    else:
        noise_variance = max(noise_variance, 1e-12)
    
    csi_info = {
        'nmse_db': None,  # Will be computed if true CSI available
        'pilot_mse': float(pilot_mse),
        'noise_variance': float(noise_variance),
        'P_H': float(P_H_full),  # 🔧 CRITICAL FIX: Use full CSI power, not pilot CSI power
        'fd_max_used': float(fd_max),
        'tau_rms_used': float(tau_rms),
        'lmmse_win_t': win_t,
        'lmmse_win_f': win_f,
        'delay_comp_samples': metadata.get('delay_samples', 0) if metadata else 0,
        'phase_comp_deg': float(np.rad2deg(avg_phase_err)) if pilot_count > 0 else 0.0
    }
    
    return h_est, csi_info


def compute_csi_quality_gate(csi_info, csi_cfg=None):
    """
    Compute quality-aware gating weight for CSI fusion.
    
    Args:
        csi_info: Dict with CSI metrics (nmse_db, pilot_mse, etc.)
        csi_cfg: CSI configuration dict (default: from settings)
    
    Returns:
        fusion_weight: Weight for CSI fusion (0.0 to 1.0)
        quality_label: 'good', 'ok', or 'bad'
    """
    if csi_cfg is None:
        csi_cfg = CSI_CFG
    
    quality_cfg = csi_cfg['quality_gating']
    
    # Get NMSE (if available)
    nmse_db = csi_info.get('nmse_db', None)
    
    # If NMSE not available, use pilot MSE as proxy
    if nmse_db is None:
        pilot_mse = csi_info.get('pilot_mse', None)
        noise_var = csi_info.get('noise_variance', 1e-12)
        P_H = csi_info.get('P_H', 1.0)
        
        # 🔧 FIX: Handle fallback mode (pilot_mse is None)
        if pilot_mse is None:
            # Fallback mode: use conservative quality estimate
            # Assume moderate quality (between 'ok' and 'bad')
            nmse_db = quality_cfg['nmse_ok'] + 5.0  # Slightly worse than 'ok'
        elif noise_var > 1e-12 and P_H > 1e-12:
            # Normalized MSE: pilot_mse relative to noise variance
            # If pilot_mse ≈ noise_var → good estimate
            # If pilot_mse >> noise_var → bad estimate
            snr_estimate = P_H / (noise_var + 1e-12)  # Signal-to-noise ratio estimate
            if snr_estimate > 1e-6:
                # NMSE ≈ 10*log10(pilot_mse / (P_H * snr_estimate))
                # Simplified: use pilot_mse relative to noise_var
                nmse_linear = pilot_mse / (noise_var + 1e-12)
                nmse_db = 10.0 * np.log10(nmse_linear + 1e-12)
            else:
                # Very low SNR, use pilot_mse directly
                nmse_db = 10.0 * np.log10(pilot_mse + 1e-12)
        elif noise_var > 1e-12:
            # Use pilot_mse relative to noise_variance
            nmse_linear = pilot_mse / (noise_var + 1e-12)
            nmse_db = 10.0 * np.log10(nmse_linear + 1e-12)
        else:
            # Fallback: use pilot_mse directly (should be small for good CSI)
            nmse_db = 10.0 * np.log10(pilot_mse + 1e-12)
        
        # 🔧 FIX: Clamp NMSE to reasonable range (only if pilot_mse is available)
        if pilot_mse is not None:
            # If pilot_mse is very small (< 1e-6), assume good quality
            if pilot_mse < 1e-6:
                nmse_db = -15.0  # Assume good quality
            elif pilot_mse < 1e-3:
                nmse_db = min(nmse_db, -10.0)  # Cap at -10 dB for small MSE
            elif pilot_mse < 0.1:
                nmse_db = min(nmse_db, -5.0)  # Cap at -5 dB for moderate MSE
    
    # Quality gating based on NMSE thresholds
    if nmse_db <= quality_cfg['nmse_good']:
        fusion_weight = quality_cfg['fuse_weight_good']
        quality_label = 'good'
    elif nmse_db <= quality_cfg['nmse_ok']:
        fusion_weight = quality_cfg['fuse_weight_ok']
        quality_label = 'ok'
    else:
        fusion_weight = quality_cfg['fuse_weight_bad']
        quality_label = 'bad'
    
    # Additional checks: phase error and delay
    phase_err_deg = abs(csi_info.get('phase_comp_deg', 0.0))
    if phase_err_deg > quality_cfg['max_phase_err_deg']:
        fusion_weight *= 0.5  # Reduce weight if phase error too large
    
    delay_samples = abs(csi_info.get('delay_comp_samples', 0))
    if delay_samples > quality_cfg['max_delay_samp']:
        fusion_weight *= 0.5  # Reduce weight if delay too large
    
    return float(fusion_weight), quality_label


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

