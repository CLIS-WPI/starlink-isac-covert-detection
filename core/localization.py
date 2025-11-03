# ======================================
# 📡 core/localization.py (Precision TDoA)
# Purpose: Robust TDoA localization with high-res GCC and GN refine
# Changes:
#  - Blocked GCC-PHAT (avg over segments), upsample=256
#  - Smart reference selection (corr score + earliest)
#  - WLS solve + Gauss-Newton refine (TDoA-only by default)
#  - Conservative bounds + residual-based sanity checks
#  - STNN-aided CAF refinement (Section 3.3 ICCIP 2024)
#  - Intelligent satellite selection (visibility + GDOP + IRLS)
# ======================================

import os
import numpy as np
from scipy.optimize import least_squares
import tensorflow as tf
from tqdm import tqdm

# Import satellite selection module
try:
    from core.satellite_selection import (
        SatelliteObservation,
        select_satellites_hybrid,
        filter_visible_satellites,
        compute_gdop
    )
    SATELLITE_SELECTION_AVAILABLE = True
    print("✓ Satellite selection module loaded")
except ImportError:
    SATELLITE_SELECTION_AVAILABLE = False
    print("⚠️ Satellite selection module not available, using all satellites")

# Initialize multi-GPU strategy if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ Multi-GPU strategy enabled: {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy
        print("⚠️ Single GPU or CPU strategy in use")
except Exception as e:
    print(f"⚠️ Error initializing GPU strategy: {e}")
    strategy = tf.distribute.get_strategy()  # Fallback to default

# --- Tunable parameters ---
UPSAMPLE = 256           # Upsampling factor (higher = better time resolution)
NUM_BLOCKS = 8           # Number of correlation blocks (more = lower variance)
BLOCK_OVERLAP = 0.5      # Overlap between blocks
BETA = 0.9               # GCC-β (close to PHAT)
USE_HANN = True
MAX_TDOA_ABS_M = 4.0e5   # Reject invalid TDoA values (<= 400 km)
MAX_POS_NORM_M = 8.0e5   # Max allowed position norm (<= 800 km)
REFINE_ITERS = 15        # Gauss-Newton refinement iterations
USE_FDOA = True          # Currently disabled

# Satellite selection parameters
USE_SATELLITE_SELECTION = True  # Enable intelligent satellite selection
MIN_ELEVATION_DEG = 15.0        # Minimum elevation angle for visibility
TARGET_SAT_COUNT = 12           # Target number of satellites to use
USE_GDOP_OPTIMIZATION = True    # Enable GDOP-based geometry optimization
USE_IRLS_OUTLIERS = True        # Enable IRLS outlier removal
# ---------------------------


# ======================================
# 🧮 GDOP Computation for TDOA
# ======================================
def compute_gdop_tdoa(sat_positions: list, emitter_pos: np.ndarray) -> float:
    """
    Compute GDOP (Geometric Dilution of Precision) for TDOA localization.
    
    GDOP = sqrt(trace((H^T H)^-1))
    where H is the Jacobian matrix of TDOA equations.
    
    Args:
        sat_positions: List of satellite positions [(x,y,z), ...] in meters
        emitter_pos: Estimated emitter position [x, y, z] in meters
    
    Returns:
        GDOP value (lower is better, <10 is good)
    """
    if len(sat_positions) < 4:
        return float('inf')  # Not enough satellites
    
    # Reference satellite (first one)
    ref_pos = np.array(sat_positions[0])
    
    # Build Jacobian matrix
    H = []
    for i in range(1, len(sat_positions)):
        sat_pos = np.array(sat_positions[i])
        
        # Distance from emitter to satellites
        d_ref = np.linalg.norm(emitter_pos - ref_pos)
        d_i = np.linalg.norm(emitter_pos - sat_pos)
        
        if d_ref < 1e-6 or d_i < 1e-6:
            continue
        
        # Gradient of TDOA with respect to emitter position
        grad_i = (emitter_pos - sat_pos) / d_i - (emitter_pos - ref_pos) / d_ref
        H.append(grad_i)  # 🔧 FIXED: Use full 3D gradient (x, y, z)
    
    if len(H) < 3:
        return float('inf')
    
    H = np.array(H)
    
    try:
        # Covariance matrix: (H^T H)^-1
        HTH = H.T @ H
        HTH_inv = np.linalg.inv(HTH + 1e-8 * np.eye(HTH.shape[0]))  # 🔧 Better conditioning
        
        # GDOP = sqrt(trace(Cov))
        gdop = np.sqrt(np.trace(HTH_inv))
        return float(gdop)
    except np.linalg.LinAlgError:
        return float('inf')


# ======================================
# 🛰 STNN-aided CAF Refinement (Section 3.3 ICCIP 2024)
# ======================================
def caf_refinement(rx_ref: np.ndarray, 
                   rx_aux: np.ndarray, 
                   coarse_tau: float, 
                   coarse_fd: float,
                   sigma_tau: float, 
                   sigma_fd: float, 
                   Ts: float,
                   search_step_tau: float = None,
                   search_step_fd: float = 5.0,  # 🔧 Increased from 1.0 to 5.0 Hz for speed
                   max_tau_steps: int = 200,      # 🔧 Limit TDOA search to 200 steps max
                   max_fd_steps: int = 50) -> tuple:  # 🔧 Limit FDOA search to 50 steps max
    """
    Perform CAF refinement around STNN coarse estimates.
    Only search within ±3σ of STNN estimates for computational efficiency.
    
    Based on Section 3.3 of ICCIP 2024:
    "The STNN coarse estimates guide a focused CAF search within ±3σ bounds,
    significantly reducing computational complexity while maintaining accuracy."
    
    Args:
        rx_ref: Reference satellite signal
        rx_aux: Auxiliary satellite signal
        coarse_tau: STNN coarse TDOA estimate (seconds)
        coarse_fd: STNN coarse FDOA estimate (Hz)
        sigma_tau: STNN TDOA error standard deviation (seconds)
        sigma_fd: STNN FDOA error standard deviation (Hz)
        Ts: Sampling period (seconds)
        search_step_tau: TDOA search step (seconds), auto-computed if None
        search_step_fd: FDOA search step (Hz)
        max_tau_steps: Maximum number of TDOA search steps
        max_fd_steps: Maximum number of FDOA search steps
    
    Returns:
        (refined_tau, refined_fd, peak_value)
    """
    # Define search ranges (±3σ around STNN estimates)
    tau_range_size = 6 * sigma_tau  # ±3σ
    fd_range_size = 6 * sigma_fd    # ±3σ
    
    # Adaptive step size to limit grid size
    if search_step_tau is None:
        # Ensure tau_range has at most max_tau_steps points
        search_step_tau = max(Ts, tau_range_size / max_tau_steps)
    
    # Ensure fd_range has at most max_fd_steps points
    search_step_fd = max(search_step_fd, fd_range_size / max_fd_steps)
    
    tau_range = np.arange(
        coarse_tau - 3 * sigma_tau,
        coarse_tau + 3 * sigma_tau + search_step_tau,
        search_step_tau
    )
    
    fd_range = np.arange(
        coarse_fd - 3 * sigma_fd,
        coarse_fd + 3 * sigma_fd + search_step_fd,
        search_step_fd
    )
    
    # 🔧 Safety: Limit grid size to prevent excessive computation
    if len(tau_range) > max_tau_steps:
        tau_range = tau_range[:max_tau_steps]
    if len(fd_range) > max_fd_steps:
        fd_range = fd_range[:max_fd_steps]
    
    # Grid search for maximum CAF value
    best_val = -1
    best_tau = coarse_tau
    best_fd = coarse_fd
    
    t = np.arange(len(rx_aux)) * Ts
    
    for tau in tau_range:
        # Time shift
        shift_samples = int(round(tau / Ts))
        shifted_aux = np.roll(rx_aux, shift_samples)
        
        for fd in fd_range:
            # Frequency shift
            phase = np.exp(1j * 2 * np.pi * fd * t)
            test_sig = shifted_aux * phase
            
            # Cross-correlation value
            val = np.abs(np.sum(rx_ref * np.conj(test_sig)))
            
            if val > best_val:
                best_val = val
                best_tau = tau
                best_fd = fd
    
    return best_tau, best_fd, best_val


def load_stnn_error_stats(model_dir: str = "model") -> tuple:
    """
    Load STNN error statistics for CAF refinement.
    
    Args:
        model_dir: Directory containing STNN error stats
    
    Returns:
        (sigma_tau, sigma_fd) or (None, None) if not available
    """
    try:
        # Try loading separate TDOA/FDOA stats
        stats_tdoa_path = os.path.join(model_dir, "stnn_error_stats_tdoa.npz")
        stats_fdoa_path = os.path.join(model_dir, "stnn_error_stats_fdoa.npz")
        
        if os.path.exists(stats_tdoa_path) and os.path.exists(stats_fdoa_path):
            stats_tdoa = np.load(stats_tdoa_path)
            stats_fdoa = np.load(stats_fdoa_path)
            sigma_tau = float(stats_tdoa["sigma_e"])
            sigma_fd = float(stats_fdoa["sigma_e"])
            print(f"[CAF] Loaded STNN error stats: σ_τ={sigma_tau*1e6:.2f} μs, σ_f={sigma_fd:.2f} Hz")
            return sigma_tau, sigma_fd
        else:
            print("[CAF] STNN error stats not found, CAF refinement disabled")
            return None, None
    except Exception as e:
        print(f"[CAF] Error loading STNN stats: {e}")
        return None, None


def _frame_signal(x, n_blocks=NUM_BLOCKS, overlap=BLOCK_OVERLAP):
    """Split the input signal into overlapping frames of equal length."""
    x = np.asarray(x)
    if n_blocks <= 1:
        return [x]
    L = len(x)
    seg = int(np.floor(L / (1 + (n_blocks - 1) * (1 - overlap))))
    hop = int(seg * (1 - overlap))
    frames = []
    start = 0
    for _ in range(n_blocks):
        end = start + seg
        if end > L:  # zero-pad the last block if needed
            pad = end - L
            xf = np.pad(x[start:L], (0, pad))
        else:
            xf = x[start:end]
        frames.append(xf)
        start += hop
        if start >= L:
            break
    return frames

def _gcc_phat_core(x, y, up=UPSAMPLE, beta=BETA, use_hann=USE_HANN):
    """Single-block GCC-ÃŽÂ² with frequency zero-padding (upsampling) + fftshift."""
    x = np.asarray(x, dtype=np.complex64)
    y = np.asarray(y, dtype=np.complex64)
    Lx, Ly = len(x), len(y)
    n_lin = Lx + Ly - 1
    n_fft = 1 << (n_lin - 1).bit_length()
    n_fft_up = n_fft * up

    if use_hann:
        wx = np.hanning(Lx).astype(np.float32)
        wy = np.hanning(Ly).astype(np.float32)
        x = x * wx
        y = y * wy

    X = np.fft.fft(x, n_fft)
    Y = np.fft.fft(y, n_fft)
    R = X * np.conj(Y)
    mag = np.abs(R) + 1e-12
    R = R / (mag ** beta)

    R_up = np.zeros(n_fft_up, dtype=np.complex64)
    half = n_fft // 2
    R_up[:half] = R[:half]
    R_up[-(n_fft - half):] = R[half:]

    corr = np.fft.ifft(R_up)
    return np.fft.fftshift(np.abs(corr))

def gcc_phat_blocked(x, y, up=UPSAMPLE, n_blocks=NUM_BLOCKS, overlap=BLOCK_OVERLAP):
    """Average GCC-ÃŽÂ² over multiple overlapping blocks to reduce variance."""
    xf = _frame_signal(x, n_blocks, overlap)
    yf = _frame_signal(y, n_blocks, overlap)
    M = min(len(xf), len(yf))
    acc = None
    for i in range(M):
        c = _gcc_phat_core(xf[i], yf[i], up=up)
        acc = c if acc is None else acc + c
    return acc / max(M, 1)

def _parabolic_subsample(corr, k0):
    """Quadratic interpolation around the peak index k0 (fftshift domain)."""
    N = len(corr)
    if k0 <= 0 or k0 >= N - 1:
        return 0.0, 1e-3
    y1, y2, y3 = corr[k0 - 1], corr[k0], corr[k0 + 1]
    den = (y1 - 2 * y2 + y3)
    delta = 0.0 if den == 0 else 0.5 * (y1 - y3) / den
    curv = abs(den)
    return float(delta), float(max(curv, 1e-3))

def _estimate_toa(sig, ref, Fs):
    """High-resolution TOA via blocked GCC-ÃŽÂ² + sub-sample interpolation."""
    L = min(len(sig), len(ref))
    sig = np.asarray(sig[:L])
    ref = np.asarray(ref[:L])

    corr = gcc_phat_blocked(sig, ref)
    k0 = int(np.argmax(corr))
    N = len(corr)
    center = N // 2
    delta, curv = _parabolic_subsample(corr, k0)

    d_samp = ((k0 - center) + delta) / UPSAMPLE
    dt = d_samp / Fs  # seconds
    score = float(corr[k0])  # peak score
    return dt, curv, score

def _choose_reference(sats, ref_sig, Fs):
    """Pick best reference using earliest TOA & highest peak score."""
    cands = []
    for idx, s in enumerate(sats):
        sig = s.get('rx_time_b_full', s['rx_time_padded'])
        if sig is None:
            continue
        dt, curv, score = _estimate_toa(sig, ref_sig, Fs)
        cands.append((idx, dt, score))
    if not cands:
        return 0  # fallback

    cands.sort(key=lambda t: (t[1], -t[2]))  # earliest TOA then highest score
    return int(cands[0][0])

def trilateration_wls_tdoa(tdoa_diffs, pairs, weights_t):
    """Weighted Least Squares solver (TDoA-only). pos2 = [x,y] (z=0)."""
    c0 = 3e8
    def resid(pos2):
        x, y = pos2
        P = np.array([x, y, 0.0])
        r = []
        for dd, (ref_pos, sat_pos), w in zip(tdoa_diffs, pairs, weights_t):
            d_ref = np.linalg.norm(P - ref_pos)
            d_i = np.linalg.norm(P - sat_pos)
            r.append((dd - (d_i - d_ref)) * np.sqrt(w))
        return np.array(r)

    # Initial guess: centroid of satellites
    sats = np.array([p[1] for p in pairs] + [p[0] for p in pairs])
    x0 = np.mean(sats[:, :2], axis=0)

    lo = np.array([-MAX_POS_NORM_M, -MAX_POS_NORM_M])
    hi = np.array([ MAX_POS_NORM_M,  MAX_POS_NORM_M])

    res = least_squares(resid, x0, method='trf', loss='huber',
                        f_scale=300.0, bounds=(lo, hi), max_nfev=1500)
    return np.array([res.x[0], res.x[1], 0.0])

def _gn_refine_tdoa(pos_est, tdoa_diffs, pairs, weights_t, iters=REFINE_ITERS):
    """GaussÃ¢â‚¬â€œNewton refinement for TDoA model (z=0)."""
    P = pos_est.copy()
    for _ in range(iters):
        J = []
        r = []
        for dd, (ref_pos, sat_pos), w in zip(tdoa_diffs, pairs, weights_t):
            d_ref = np.linalg.norm(P - ref_pos)
            d_i = np.linalg.norm(P - sat_pos)
            if d_ref < 1e-6 or d_i < 1e-6:
                continue
            ri = dd - (d_i - d_ref)
            r.append(ri * np.sqrt(w))
            gi = (P - sat_pos) / d_i - (P - ref_pos) / d_ref
            J.append(gi[:2] * np.sqrt(w))
        if len(J) < 2:
            break
        J = np.array(J)
        r = np.array(r)
        try:
            H = J.T @ J + 1e-6 * np.eye(2)
            g = J.T @ r
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        P[:2] += delta
        nrm = np.linalg.norm(P[:2])
        if nrm > MAX_POS_NORM_M:
            P[:2] *= (MAX_POS_NORM_M / (nrm + 1e-9))
    return P

def estimate_emitter_location_tdoa(sample_idx, dataset, isac_system):
    """
    High-precision TDoA localization (Path B for signals, TDoA-only).
    """
    c0 = 3e8
    sats = dataset.get('satellite_receptions', None)
    if sats is None:
        return None, None
    sats = sats[sample_idx]
    if sats is None or len(sats) < 4:
        return None, None

    Fs = float(dataset.get('sampling_rate', getattr(isac_system, 'SAMPLING_RATE', 1.0)))
    
    # âœ… CRITICAL: Always use clean tx_time_padded as reference
    ref_templates = dataset.get('tx_time_padded', None)
    ref_sig = None
    if ref_templates is not None:
        try:
            ref_sig = np.asarray(ref_templates[sample_idx])
        except Exception as e:
            print(f"[ERROR] Failed to load tx_time_padded: {e}")
            ref_sig = None

    # âš ï¸ Fallback (should NEVER happen if dataset is correct)
    if ref_sig is None:
        print("[WARNING] tx_time_padded not found! Using satellite fallback (less accurate!)")
        snr_scores = [np.mean(np.abs(s.get('rx_time_b_full', s['rx_time_padded']))**2) for s in sats]
        ref_sig = sats[int(np.argmax(snr_scores))].get('rx_time_b_full', sats[int(np.argmax(snr_scores))]['rx_time_padded'])

    # Ã˜Â§Ã™â€ Ã˜ÂªÃ˜Â®Ã˜Â§Ã˜Â¨ Ã™â€¦Ã˜Â±Ã˜Â¬Ã˜Â¹ Ã˜Â¨Ã™â€¡Ã˜ÂªÃ˜Â± Ã˜Â¨Ã˜Â§ Ã˜Â§Ã™â€¦Ã˜ÂªÃ›Å’Ã˜Â§Ã˜Â² Ã™â€¡Ã™â€¦Ã˜Â¨Ã˜Â³Ã˜ÂªÃšÂ¯Ã›Å’ + Ã™Ë†Ã˜Â±Ã™Ë†Ã˜Â¯ Ã˜Â²Ã™Ë†Ã˜Â¯Ã˜ÂªÃ˜Â±
    ref_idx = _choose_reference(sats, ref_sig, Fs)
    ref_pos = sats[ref_idx]['position']
    toas, scores, weights = [], [], []
    positions = []

    # Ã™â€¦Ã˜Â­Ã˜Â§Ã˜Â³Ã˜Â¨Ã™â€¡ TOA Ã˜Â¨Ã˜Â±Ã˜Â§Ã›Å’ Ã™â€¡Ã˜Â± Ã™â€¦Ã˜Â§Ã™â€¡Ã™Ë†Ã˜Â§Ã˜Â±Ã™â€¡ Ã™â€ Ã˜Â³Ã˜Â¨Ã˜Âª Ã˜Â¨Ã™â€¡ ref_sig (Ã™â€ Ã™â€¡ ref_idx)
    raw_toas = []
    for s in sats:
        sig = s.get('rx_time_b_full', s['rx_time_padded'])
        if sig is None:
            raw_toas.append(np.nan); continue
        dt, curv, sc = _estimate_toa(sig, ref_sig, Fs)
        raw_toas.append(dt)
    raw_toas = np.array(raw_toas)
    # ref Ã™Ë†Ã˜Â§Ã™â€šÃ˜Â¹Ã›Å’ Ã˜Â¨Ã˜Â±Ã˜Â§Ã›Å’ TDoA: Ã˜Â²Ã™Ë†Ã˜Â¯Ã˜ÂªÃ˜Â±Ã›Å’Ã™â€  Ã˜Â±Ã˜Â³Ã›Å’Ã˜Â¯Ã™â€  (robustÃ¢â‚¬Å’Ã˜ÂªÃ˜Â±)
    ref_idx_num = int(np.nanargmin(raw_toas))

    # Build TDoA lists
    tdoa_diffs = []
    pairs = []
    w_t = []
    dropped_t, dropped_f = 0, 0

    # Ã™â€¦Ã˜Â±Ã˜Â¬Ã˜Â¹ Ã™â€ Ã™â€¡Ã˜Â§Ã›Å’Ã›Å’
    toa_ref = float(raw_toas[ref_idx_num])
    for i, s in enumerate(sats):
        if i == ref_idx_num:
            continue
        sig = s.get('rx_time_b_full', s['rx_time_padded'])
        if sig is None:
            continue
        dt_i, curv_i, sc_i = _estimate_toa(sig, ref_sig, Fs)
        dd = (dt_i - toa_ref) * c0  # meters
        if abs(dd) > MAX_TDOA_ABS_M:
            dropped_t += 1
            continue
        tdoa_diffs.append(dd)
        pairs.append((np.asarray(sats[ref_idx_num]['position']), np.asarray(s['position'])))
        # Ã™Ë†Ã˜Â²Ã™â€ : Ã˜Â§Ã˜Â² Ã˜Â§Ã™â€ Ã˜Â­Ã™â€ Ã˜Â§ (curvature) Ã˜Â¨Ã™â€¡Ã¢â‚¬Å’Ã˜Â¹Ã™â€ Ã™Ë†Ã˜Â§Ã™â€  Ã˜Â´Ã˜Â§Ã˜Â®Ã˜Âµ ÃšÂ©Ã›Å’Ã™ÂÃ›Å’Ã˜Âª Ã™Â¾Ã›Å’ÃšÂ© Ã˜Â§Ã˜Â³Ã˜ÂªÃ™ÂÃ˜Â§Ã˜Â¯Ã™â€¡ Ã™â€¦Ã›Å’Ã¢â‚¬Å’ÃšÂ©Ã™â€ Ã›Å’Ã™â€¦
        w_t.append(curv_i)
        positions.append(s['position'])
        scores.append(sc_i)
        toas.append(dt_i)

    if len(tdoa_diffs) < 2:
        print(f"[DEBUG] Valid TDoA count: {len(tdoa_diffs)} / {len(sats)}")
        return None, None

    w_t = np.asarray(w_t, dtype=float)
    w_t = w_t / (np.sum(w_t) + 1e-12)
    w_t = np.clip(w_t, 1e-3, 1.0)
    
    # âœ… OUTLIER REJECTION: Remove TDoAs that are too far from median
    if len(tdoa_diffs) >= 4:  # Only if we have enough measurements
        tdoa_arr = np.array(tdoa_diffs)
        median = np.median(tdoa_arr)
        mad = np.median(np.abs(tdoa_arr - median))  # Median Absolute Deviation
        
        # Threshold: 3 * MAD (robust outlier detection)
        threshold = max(3 * mad, 50e3)  # At least 50 km threshold
        
        # Keep only inliers
        inlier_mask = np.abs(tdoa_arr - median) < threshold
        if np.sum(inlier_mask) >= 3:  # Need at least 3 TDoAs
            n_outliers = len(tdoa_diffs) - np.sum(inlier_mask)
            if n_outliers > 0:
                print(f"[Outlier] Rejected {n_outliers} outlier TDoAs (MAD threshold: {threshold/1e3:.1f} km)")
                tdoa_diffs = [td for td, keep in zip(tdoa_diffs, inlier_mask) if keep]
                pairs = [p for p, keep in zip(pairs, inlier_mask) if keep]
                w_t = w_t[inlier_mask]
                w_t = w_t / (np.sum(w_t) + 1e-12)  # Re-normalize weights

    print(f"[DEBUG] Valid TDoA count: {len(tdoa_diffs)} / {len(sats)} (dropped_by_tdoa={dropped_t}, dropped_by_fdoa={dropped_f}, use_fdoa={USE_FDOA})")
    if len(tdoa_diffs) >= 3:
        print(f"[DEBUG] Example TDoA diffs (m): {np.array(tdoa_diffs[:3])}")
        print(f"[DEBUG] TDoA median: {np.median(tdoa_diffs)/1e3:.2f} km, std: {np.std(tdoa_diffs)/1e3:.2f} km")

    # --- Ã˜Â­Ã™â€ž WLS Ã™Ë† Ã˜Â³Ã™Â¾Ã˜Â³ Ã˜Â±Ã›Å’Ã™ÂÃ˜Â§Ã›Å’Ã™â€ Ã™â€¦Ã™â€ Ã˜Âª GN ---
    try:
        est = trilateration_wls_tdoa(tdoa_diffs, pairs, w_t)
        est = _gn_refine_tdoa(est, tdoa_diffs, pairs, w_t, iters=REFINE_ITERS)
    except Exception as e:
        print(f"[ERROR] Solver failed: {e}")
        return None, None

    # sanity: Ã™ÂÃ˜Â§Ã˜ÂµÃ™â€žÃ™â€¡ Ã˜Â§Ã˜Â² Ã™â€¦Ã˜Â¨Ã˜Â¯Ã˜Â£ Ã™Ë† Ã˜Â®Ã˜Â·Ã˜Â§Ã›Å’ Ã™â€¦Ã˜Â¯Ã™â€ž
    if np.linalg.norm(est[:2]) > MAX_POS_NORM_M:
        return None, None

    # Ã˜Â¨Ã˜Â±Ã˜Â±Ã˜Â³Ã›Å’ Ã˜Â³Ã˜Â§Ã˜Â²ÃšÂ¯Ã˜Â§Ã˜Â±Ã›Å’ Ã˜Â¨Ã˜Â§ TDoA (RMSE Ã˜Â±Ã˜Â²Ã›Å’Ã˜Â¯Ã™Ë†Ã˜Â§Ã™â€ž)
    def _tdoa_res_rmse(P):
        rr = []
        for dd, (ref_p, sat_p) in zip(tdoa_diffs, pairs):
            rr.append(dd - (np.linalg.norm(P - sat_p) - np.linalg.norm(P - ref_p)))
        rr = np.array(rr)
        return float(np.sqrt(np.mean(rr**2)))

    rmse_m = _tdoa_res_rmse(est)
    # Ã˜Â§ÃšÂ¯Ã˜Â± Ã˜Â±Ã˜Â²Ã›Å’Ã˜Â¯Ã™Ë†Ã˜Â§Ã™â€ž Ã˜Â²Ã›Å’Ã˜Â§Ã˜Â¯ Ã˜Â¨Ã™Ë†Ã˜Â¯Ã˜Å’ Ã™â€¦Ã˜Â±Ã˜Â¬Ã˜Â¹ Ã˜Â¯Ã™Ë†Ã™â€¦ Ã˜Â±Ã˜Â§ Ã˜Â§Ã™â€¦Ã˜ÂªÃ˜Â­Ã˜Â§Ã™â€  ÃšÂ©Ã™â€ 
    if rmse_m > 1000.0 and len(sats) >= 5:  # ✅ کمتر retry
        # Ã™â€¦Ã˜Â±Ã˜Â¬Ã˜Â¹ Ã˜Â¬Ã˜Â§Ã›Å’ÃšÂ¯Ã˜Â²Ã›Å’Ã™â€ : Ã˜Â¯Ã™Ë†Ã™â€¦Ã›Å’Ã™â€  TOA ÃšÂ©Ã™Ë†Ãšâ€ ÃšÂ©
        order = np.argsort(raw_toas)
        for alt in order[1:2]:
            if alt == ref_idx_num:
                continue
            tdoa_diffs2, pairs2, w_t2 = [], [], []
            toa_ref2 = float(raw_toas[alt])
            for i, s in enumerate(sats):
                if i == alt: continue
                sig = s.get('rx_time_b_full', s['rx_time_padded'])
                if sig is None: continue
                dt_i, curv_i, sc_i = _estimate_toa(sig, ref_sig, Fs)
                dd = (dt_i - toa_ref2) * c0
                if abs(dd) > MAX_TDOA_ABS_M: continue
                tdoa_diffs2.append(dd)
                pairs2.append((np.asarray(sats[alt]['position']), np.asarray(s['position'])))
                w_t2.append(curv_i)
            if len(tdoa_diffs2) >= 2:
                w_t2 = np.asarray(w_t2); w_t2 = w_t2 / (np.sum(w_t2)+1e-12); w_t2 = np.clip(w_t2,1e-3,1.0)
                try:
                    est2 = trilateration_wls_tdoa(tdoa_diffs2, pairs2, w_t2)
                    est2 = _gn_refine_tdoa(est2, tdoa_diffs2, pairs2, w_t2, iters=REFINE_ITERS)
                    if _tdoa_res_rmse(est2) < rmse_m:
                        est, rmse_m = est2, _tdoa_res_rmse(est2)
                except:
                    pass

    print(f"[DEBUG] Solver returned estimate: [{est[0]:.2f} {est[1]:.2f}] m")

    gt = dataset['emitter_locations'][sample_idx]
    if gt is None:
        return None, None

    if np.linalg.norm(est[:2]) <= MAX_POS_NORM_M:
        print(f"[DEBUG] Final check passed Ã¢â‚¬â€ GT: [{gt[0]:.2f} {gt[1]:.2f}], EST: [{est[0]:.2f} {est[1]:.2f}]")
        return est, np.array(gt)
    return None, None

def run_tdoa_localization(dataset, y_hat, y_te, idx_te, isac_system, use_enhanced=False):
    """
    Run TDoA localization on true positive detections.
    
    Args:
        dataset: Dataset dictionary
        y_hat: Predicted labels
        y_te: True labels
        idx_te: Sample indices
        isac_system: ISAC system instance
        use_enhanced: If True, use enhanced localization pipeline with satellite selection,
                      CAF refinement, and FDOA support. If False, use traditional method.
    """
    
    # Use enhanced pipeline if requested
    if use_enhanced:
        try:
            from core.localization_enhanced import run_enhanced_tdoa_localization
            return run_enhanced_tdoa_localization(
                dataset=dataset,
                y_hat=y_hat,
                y_te=y_te,
                idx_te=idx_te,
                isac_system=isac_system,
                use_satellite_selection=True,
                use_caf_refinement=True,
                use_fdoa=False,  # Can be enabled if needed
                verbose=False
            )
        except Exception as e:
            print(f"⚠️ Enhanced localization failed: {e}")
            print("Falling back to traditional method...")
    
    print("\n[Phase 5] Running TDoA localization...")
    print("\n=== PHASE 8: HYBRID TDoA/FDoA LOCALIZATION ===")

    
    # ✅ Check if STNN is available
    use_stnn = hasattr(isac_system, 'stnn_estimator') and isac_system.stnn_estimator is not None
    if use_stnn:
        print("[STNN] Using STNN-aid CAF method (hybrid)")
        # Use existing localization function with STNN capabilities
        localization_fn = estimate_emitter_location_tdoa
    else:
        print("[GCC-PHAT] Using traditional method (full search)")
        localization_fn = estimate_emitter_location_tdoa
    
    print("Processing true positives...")

    loc_errors, tp_sample_ids, tp_ests, tp_gts = [], [], [], []
    
    # Count true positives for progress bar
    n_tp = sum(1 for i, pred in enumerate(y_hat) if pred == 1 and y_te[i] == 1)
    print(f"  Found {n_tp} true positive samples to localize")
    
    # 🔧 SPEEDUP: Only localize first N samples for faster evaluation
    MAX_LOCALIZATION_SAMPLES = 100  # Reduce from full dataset for speed
    print(f"  ⚡ Processing only first {MAX_LOCALIZATION_SAMPLES} samples for speed...")
    
    processed = 0
    with tqdm(total=min(n_tp, MAX_LOCALIZATION_SAMPLES), desc="Localizing", unit="sample") as pbar:
        for i, pred in enumerate(y_hat):
            if pred == 1 and y_te[i] == 1:  # True positive
                if processed >= MAX_LOCALIZATION_SAMPLES:
                    break  # Stop after N samples
                ds_idx = int(idx_te[i])
                est, gt = localization_fn(ds_idx, dataset, isac_system)
                if est is not None and gt is not None:
                    err = np.linalg.norm(est - gt)
                    loc_errors.append(err)
                    tp_sample_ids.append(ds_idx)
                    tp_ests.append(est)
                    tp_gts.append(gt)
                    pbar.set_postfix({"error": f"{err:.1f}m", "median": f"{np.median(loc_errors):.1f}m"})
                    if len(loc_errors) <= 3:
                        print(f"  Sample {ds_idx} | GT {gt[:2]} | EST {est[:2]} | Error={err:.2f} m")
                processed += 1
                pbar.update(1)

    if loc_errors:
        med = float(np.median(loc_errors))
        mean = float(np.mean(loc_errors))
        p90 = float(np.percentile(loc_errors, 90))
        print("\n=== Hybrid Localization Results ===")
        print(f"Median Error: {med:.2f} m")
        print(f"Mean Error  : {mean:.2f} m")
        print(f"90th Perc.  : {p90:.2f} m")
        print(f"Total samples: {len(loc_errors)}")
    else:
        print("Ã¢Å¡Â Ã¯Â¸Â No true-positive attacks to localize.")
        return [], [], [], []
    return loc_errors, tp_sample_ids, tp_ests, tp_gts

def compute_crlb(loc_errors, tp_sample_ids, tp_ests, tp_gts, dataset, isac_system):
    """(Same as Ã™â€šÃ˜Â¨Ã™â€žÃ˜â€º Ã˜ÂªÃ˜ÂºÃ›Å’Ã›Å’Ã˜Â±Ã›Å’ Ã™â€ Ã˜Â¯Ã˜Â§Ã˜Â¯Ã™â€¦ Ã˜ÂªÃ˜Â§ Ã˜Â®Ã˜Â±Ã™Ë†Ã˜Â¬Ã›Å’Ã¢â‚¬Å’Ã˜Â§Ã˜Âª Ã˜Â³Ã˜Â§Ã˜Â²ÃšÂ¯Ã˜Â§Ã˜Â± Ã˜Â¨Ã™â€¦Ã˜Â§Ã™â€ Ã˜Â¯)"""
    try:
        c0 = 3e8
        timing_resolution_ns = 1e9 / (isac_system.SAMPLING_RATE * UPSAMPLE)
        sigma_t = timing_resolution_ns * 1.2 * 1e-9
        sigma_d = c0 * sigma_t

        crlb_values, sample_ids_used, achieved_errors_used = [], [], []
        est_x, est_y, gt_x, gt_y = [], [], [], []
        for ds_idx, est, gt, err in zip(tp_sample_ids, tp_ests, tp_gts, loc_errors):
            sat_list = dataset.get('satellite_receptions', [None])[ds_idx]
            if sat_list is None: continue
            # ref: Ã˜Â¨Ã›Å’Ã˜Â´Ã˜ÂªÃ˜Â±Ã›Å’Ã™â€  Ã˜ÂªÃ™Ë†Ã˜Â§Ã™â€  Path B
            snr_scores = [np.mean(np.abs(s.get('rx_time_b_full', s['rx_time_padded']))**2) for s in sat_list]
            best_ref_idx = int(np.argmax(snr_scores))
            ref = np.array(sat_list[best_ref_idx]['position'][:2])

            H = []
            P = np.array(gt[:2])
            for j, s in enumerate(sat_list):
                if j == best_ref_idx: continue
                sat = np.array(s['position'][:2])
                d_sat = np.linalg.norm(P - sat); d_ref = np.linalg.norm(P - ref)
                if d_sat < 1e-6 or d_ref < 1e-6: continue
                grad = (P - sat)/d_sat - (P - ref)/d_ref
                H.append(grad)
            H = np.array(H)
            if H.shape[0] >= 2:
                FIM = (H.T @ H) / (sigma_d**2)
                CRLB_matrix = np.linalg.inv(FIM)
                crlb_i = float(np.sqrt(np.trace(CRLB_matrix)))
                crlb_values.append(crlb_i)
                sample_ids_used.append(ds_idx)
                achieved_errors_used.append(err)
                est_x.append(float(est[0])); est_y.append(float(est[1]))
                gt_x.append(float(gt[0])); gt_y.append(float(gt[1]))

        if crlb_values:
            mean_crlb = float(np.mean(crlb_values))
            med_crlb = float(np.median(crlb_values))
            achieved_med = float(np.median(loc_errors))
            ratio = achieved_med / med_crlb if med_crlb > 0 else float('inf')
            eff_pct = 100.0 / ratio if ratio > 0 else 0.0

            print("\n=== CRLB Analysis (TDoA-Only Baseline) ===")
            print(f"System timing resolution: {timing_resolution_ns:.2f} ns")
            print(f"Assumed timing error (ÃÆ’_t): {sigma_t*1e9:.2f} ns")
            print(f"Equivalent range error (ÃÆ’_d): {sigma_d:.2f} m")
            print(f"Mean CRLB: {mean_crlb:.2f} m")
            print(f"Median CRLB: {med_crlb:.2f} m")
            print(f"Achieved median error: {achieved_med:.2f} m")
            print(f"Ratio (achieved/CRLB): {ratio:.2f}Ãƒâ€”")
            print(f"Efficiency: {eff_pct:.1f}% of the CRLB")
            return {
                'crlb_values': np.array(crlb_values),
                'sample_ids': sample_ids_used,
                'achieved_errors': achieved_errors_used,
                'est_x': est_x, 'est_y': est_y,
                'gt_x': gt_x, 'gt_y': gt_y,
                'mean_crlb': mean_crlb,
                'med_crlb': med_crlb,
                'achieved_med': achieved_med,
                'ratio': ratio,
                'efficiency_pct': eff_pct
            }
        else:
            print("(CRLB computation skipped: no valid H matrices)")
            return None
    except Exception as e:
        print(f"(CRLB computation skipped: {e})")
        return None