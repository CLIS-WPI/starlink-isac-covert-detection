# ======================================
# 📄 core/localization.py
# Purpose: TDoA-based localization with GCC-PHAT, WLS, and CRLB analysis
# ======================================

import numpy as np
from scipy.optimize import least_squares
from config.settings import ABLATION_CONFIG


def gcc_phat(x, y, upsample_factor=32, beta=0.5, use_hann=True):
    """
    GCC-PHAT with frequency-domain upsampling.
    
    Args:
        x, y: Complex baseband signals
        upsample_factor: Upsampling factor (8/16/32)
        beta: GCC-β parameter (1.0=PHAT, <1 better at low SNR)
        use_hann: Apply Hann window
    
    Returns:
        ndarray: Cross-correlation with fftshift applied
    """
    x = np.asarray(x, dtype=np.complex64)
    y = np.asarray(y, dtype=np.complex64)
    
    Lx, Ly = len(x), len(y)
    n_lin = Lx + Ly - 1
    n_fft = 1 << (n_lin - 1).bit_length()
    n_fft_up = n_fft * upsample_factor
    
    # Apply Hann window
    if use_hann:
        wx = np.hanning(Lx).astype(np.float32)
        wy = np.hanning(Ly).astype(np.float32)
        x = x * wx
        y = y * wy
    
    # FFT
    X = np.fft.fft(x, n_fft)
    Y = np.fft.fft(y, n_fft)
    R = X * np.conj(Y)
    
    # GCC-β weighting
    mag = np.abs(R) + 1e-12
    R = R / (mag ** beta)
    
    # Frequency-domain zero-padding (upsampling)
    R_up = np.zeros(n_fft_up, dtype=np.complex64)
    half = n_fft // 2
    R_up[:half] = R[:half]
    R_up[-(n_fft - half):] = R[half:]
    
    # IFFT + fftshift
    corr = np.fft.ifft(R_up)
    corr = np.fft.fftshift(np.abs(corr))
    
    # Use multithreaded FFT backend if available
    try:
        np.fft.set_fft_backend("pocketfft")
    except:
        pass
    
    return corr


def trilateration_2d_wls(tdoa_diffs, pairs, weights):
    """
    Weighted Least Squares (WLS) 2D trilateration with Huber loss.
    
    Args:
        tdoa_diffs: TDoA measurements (range differences in meters)
        pairs: List of (reference_pos, satellite_pos) tuples
        weights: Weights for each measurement
    
    Returns:
        ndarray: Estimated position [x, y, 0]
    """
    def resid_wls(pos2, tdoa_list, pairs, weights):
        x, y = pos2
        P = np.array([x, y, 0.0])
        r = []
        for dd, (ref_pos, sat_pos), w in zip(tdoa_list, pairs, weights):
            d_ref = np.linalg.norm(P - ref_pos)
            d_i = np.linalg.norm(P - sat_pos)
            r.append((dd - (d_i - d_ref)) * np.sqrt(w))
        return np.array(r)
    
    # Initial guess: centroid of satellites
    sat_positions = np.array([pair[0] for pair in pairs])
    x0 = np.mean(sat_positions[:, :2], axis=0)
    
    # Bounds
    lo, hi = np.array([-2e5, -2e5]), np.array([2e5, 2e5])
    
    # Solve with Huber loss (robust to outliers)
    res = least_squares(
        resid_wls, x0,
        args=(tdoa_diffs, pairs, weights),
        method='trf',
        loss='huber',
        f_scale=300.0,
        bounds=(lo, hi),
        max_nfev=2000
    )
    
    return np.array([res.x[0], res.x[1], 0.0])


def estimate_emitter_location_tdoa(sample_idx, dataset, isac_system):
    """
    Estimate emitter location using TDoA with GCC-PHAT and WLS.
    
    Args:
        sample_idx: Sample index in dataset
        dataset: Full dataset dictionary
        isac_system: ISACSystem instance
    
    Returns:
        tuple: (estimated_position, ground_truth_position)
    """
    sat_data = dataset.get('satellite_receptions', None)
    if sat_data is None:
        return None, None
    
    sats = sat_data[sample_idx]
    if sats is None or len(sats) < 4:
        return None, None
    
    # Use clean transmit template (Path B)
    tx_templates = dataset.get('tx_time_padded', None)
    ref_sig = None
    if tx_templates is not None:
        try:
            ref_sig = np.asarray(tx_templates[sample_idx])
        except:
            ref_sig = None
    
    # Fallback: best-SNR satellite
    if ref_sig is None:
        snr_scores = [
            np.mean(np.abs(s.get('rx_time_padded', s['rx_time']))**2) 
            for s in sats
        ]
        best_ref_idx = int(np.argmax(snr_scores))
        ref = sats[best_ref_idx]
        ref_sig = ref.get('rx_time_padded', ref['rx_time'])
        ref_pos = ref['position']
    
    c0 = 3e8
    Fs = float(dataset.get('sampling_rate', isac_system.SAMPLING_RATE))
    upsampling_factor = 32
    
    # Compute TOA per satellite
    toas = []
    positions = []
    corr_weights = []
    
    for s in sats:
        pos = s['position']
        positions.append(pos)
        
        # Get signal (prefer Path B: rx_time_padded)
        sig = np.asarray(s.get('rx_time_padded', s['rx_time']))
        
        L = min(len(sig), len(ref_sig))
        sig = np.asarray(sig[:L])
        re = np.asarray(ref_sig[:L])
        
        # GCC-PHAT correlation
        corr = gcc_phat(sig, re, upsample_factor=upsampling_factor, 
                       beta=1.0, use_hann=False)
        center = len(corr) // 2
        k0 = int(np.argmax(np.abs(corr)))
        
        # Parabolic interpolation
        if 0 < k0 < len(corr) - 1:
            y1, y2, y3 = np.abs(corr[k0-1]), np.abs(corr[k0]), np.abs(corr[k0+1])
            den = (y1 - 2*y2 + y3)
            delta = 0.0 if den == 0 else 0.5*(y1 - y3)/den
            curv = abs(den)
            weight = (max(curv, 1e-6) 
                     if ABLATION_CONFIG.get('use_curvature_weights', True) 
                     else 1.0)
        else:
            delta = 0.0
            weight = (1e-6 
                     if ABLATION_CONFIG.get('use_curvature_weights', True) 
                     else 1.0)
        
        # Convert to time
        d_samp = ((k0 - center) + delta) / upsampling_factor
        dt = d_samp / Fs
        toas.append(dt)
        corr_weights.append(weight)
    
    if len(toas) < 2:
        return None, None
    
    toas = np.array(toas)
    corr_weights = np.array(corr_weights)
    
    # Choose reference: earliest arrival (minimum TOA)
    ref_idx_num = int(np.argmin(toas))
    toa_ref = float(toas[ref_idx_num])
    ref_pos = positions[ref_idx_num]
    
    # Build TDoA differences (range differences in meters)
    tdoa_diffs = []
    pairs = []
    weights = []
    for i, (toa_i, pos_i, w) in enumerate(zip(toas, positions, corr_weights)):
        if i == ref_idx_num:
            continue
        dd = (float(toa_i) - toa_ref) * c0
        tdoa_diffs.append(dd)
        pairs.append((ref_pos, pos_i))
        weights.append(w)
    
    if len(tdoa_diffs) < 2:
        return None, None
    
    # Normalize weights
    weights = np.array(weights)
    weights = np.clip(weights / np.max(weights), 1e-3, 1.0)
    
    # WLS trilateration
    try:
        est = trilateration_2d_wls(tdoa_diffs, pairs, weights)
    except:
        return None, None
    
    # Ground truth
    gt = dataset['emitter_locations'][sample_idx]
    if gt is None:
        return None, None
    
    return est, np.array(gt)


def run_tdoa_localization(dataset, y_hat, y_te, idx_te, isac_system):
    """
    Run TDoA localization on true positive samples.
    
    Args:
        dataset: Full dataset dictionary
        y_hat: Predictions (binary)
        y_te: True labels
        idx_te: Test indices
        isac_system: ISACSystem instance
    
    Returns:
        tuple: (loc_errors, sample_ids, estimates, ground_truths)
    """
    print("\n=== PHASE 8: TDoA-BASED LOCALIZATION ===")
    print("Processing true positives...")
    
    loc_errors = []
    tp_sample_ids = []
    tp_ests = []
    tp_gts = []
    
    for i, pred in enumerate(y_hat):
        if pred == 1 and y_te[i] == 1:  # True positive
            ds_idx = int(idx_te[i])
            est, gt = estimate_emitter_location_tdoa(ds_idx, dataset, isac_system)
            if est is not None and gt is not None:
                err = np.linalg.norm(est - gt)
                loc_errors.append(err)
                tp_sample_ids.append(ds_idx)
                tp_ests.append(est)
                tp_gts.append(gt)
                if len(loc_errors) <= 3:
                    print(f"  Sample {ds_idx} | GT {gt[:2]} | EST {est[:2]} | Error={err:.2f} m")
    
    if loc_errors:
        med = float(np.median(loc_errors))
        mean = float(np.mean(loc_errors))
        p90 = float(np.percentile(loc_errors, 90))
        
        print("\n=== TDoA Localization Results ===")
        print(f"Median Error: {med:.2f} m")
        print(f"Mean Error  : {mean:.2f} m")
        print(f"90th Perc.  : {p90:.2f} m")
        print(f"Total samples: {len(loc_errors)}")
    else:
        print("⚠️ No true-positive attacks to localize.")
        return [], [], [], []
    
    return loc_errors, tp_sample_ids, tp_ests, tp_gts


def compute_crlb(loc_errors, tp_sample_ids, tp_ests, tp_gts, dataset, isac_system):
    """
    Compute Cramér-Rao Lower Bound (CRLB) for TDoA localization.
    
    Args:
        loc_errors: Localization errors
        tp_sample_ids: True positive sample IDs
        tp_ests: Estimated positions
        tp_gts: Ground truth positions
        dataset: Full dataset
        isac_system: ISACSystem instance
    
    Returns:
        dict: CRLB analysis results
    """
    import os
    
    try:
        c0 = 3e8
        upsampling_factor = 32
        timing_resolution_ns = 1e9 / (isac_system.SAMPLING_RATE * upsampling_factor)
        sigma_t = timing_resolution_ns * 1.2 * 1e-9
        sigma_d = c0 * sigma_t
        
        # Control CRLB sample count
        CRLB_SAMPLES = os.getenv('CRLB_SAMPLES', 'all')
        if CRLB_SAMPLES != 'all':
            CRLB_SAMPLES = int(CRLB_SAMPLES)
        
        if isinstance(CRLB_SAMPLES, str) and CRLB_SAMPLES == 'all':
            use_idxs = list(range(len(tp_sample_ids)))
        else:
            use_idxs = list(range(min(int(CRLB_SAMPLES), len(tp_sample_ids))))
        
        crlb_values = []
        sample_ids_used = []
        achieved_errors_used = []
        est_x, est_y = [], []
        gt_x, gt_y = [], []
        
        for ii in use_idxs:
            ds_idx = tp_sample_ids[ii]
            sat_list = dataset.get('satellite_receptions', [None])[ds_idx]
            if sat_list is None:
                continue
            
            try:
                # Find reference satellite
                snr_scores = [
                    np.mean(np.abs(s.get('rx_time_padded', s['rx_time']))**2) 
                    for s in sat_list
                ]
                best_ref_idx = int(np.argmax(snr_scores))
                ref = sat_list[best_ref_idx]['position']
                
                # Build Jacobian matrix H
                H = []
                for s in sat_list:
                    if s['satellite_id'] == best_ref_idx:
                        continue
                    sat = np.array(s['position'][:2])
                    ref2 = np.array(ref[:2])
                    
                    # Evaluation point P
                    try:
                        P = np.array(tp_gts[ii][:2])
                    except:
                        P = np.array(tp_ests[ii][:2])
                    
                    d_sat = np.linalg.norm(P - sat)
                    d_ref = np.linalg.norm(P - ref2)
                    
                    if d_sat < 1e-6 or d_ref < 1e-6:
                        continue
                    
                    # Gradient of (||P-sat|| - ||P-ref||) wrt P
                    grad = (P - sat) / d_sat - (P - ref2) / d_ref
                    H.append(grad)
                
                H = np.array(H)
                if H.shape[0] >= 2:
                    # Fisher Information Matrix
                    FIM = (H.T @ H) / (sigma_d**2)
                    CRLB_matrix = np.linalg.inv(FIM)
                    crlb_i = float(np.sqrt(np.trace(CRLB_matrix)))
                    
                    crlb_values.append(crlb_i)
                    sample_ids_used.append(ds_idx)
                    achieved_errors_used.append(loc_errors[ii])
                    est_x.append(float(tp_ests[ii][0]))
                    est_y.append(float(tp_ests[ii][1]))
                    gt_x.append(float(tp_gts[ii][0]))
                    gt_y.append(float(tp_gts[ii][1]))
            except:
                continue
        
        if crlb_values:
            mean_crlb = float(np.mean(crlb_values))
            med_crlb = float(np.median(crlb_values))
            achieved_med = float(np.median(loc_errors))
            
            if med_crlb > 0:
                ratio = achieved_med / med_crlb
                eff_pct = 100.0 / ratio if ratio > 0 else 0.0
            else:
                ratio = float('inf')
                eff_pct = 0.0
            
            print("\n=== CRLB Analysis ===")
            print(f"System timing resolution: {timing_resolution_ns:.2f} ns")
            print(f"Assumed timing error (σ_t): {sigma_t*1e9:.2f} ns")
            print(f"Equivalent range error (σ_d): {sigma_d:.2f} m")
            print(f"Mean CRLB: {mean_crlb:.2f} m")
            print(f"Median CRLB: {med_crlb:.2f} m")
            print(f"Achieved median error: {achieved_med:.2f} m")
            print(f"Ratio (achieved/CRLB): {ratio:.2f}×")
            print(f"Efficiency: {eff_pct:.1f}% of the CRLB")
            
            return {
                'crlb_values': np.array(crlb_values),
                'sample_ids': sample_ids_used,
                'achieved_errors': achieved_errors_used,
                'est_x': est_x,
                'est_y': est_y,
                'gt_x': gt_x,
                'gt_y': gt_y,
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