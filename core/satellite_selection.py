# ======================================
# 📡 core/satellite_selection.py
# Purpose: Intelligent satellite selection for localization
# Features:
#  - Visibility filtering (elevation angle)
#  - Coarse position estimation
#  - Residual-based outlier detection
#  - GDOP-based geometric optimization
#  - IRLS (Iteratively Reweighted Least Squares)
#  - RANSAC for robust estimation
# ======================================

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SatelliteObservation:
    """Single satellite observation data."""
    id: int
    position: np.ndarray  # (3,) ECEF position [m]
    velocity: np.ndarray  # (3,) ECEF velocity [m/s]
    rx_signal: np.ndarray  # Received signal
    obs_tdoa: Optional[float] = None  # Observed TDOA [s]
    obs_fdoa: Optional[float] = None  # Observed FDOA [Hz]
    snr: Optional[float] = None  # Signal-to-noise ratio [dB]
    rx_power: Optional[float] = None  # Received power


def compute_elevation_angle(sat_pos: np.ndarray, ground_pos: np.ndarray) -> float:
    """
    Compute elevation angle from ground position to satellite.
    
    Args:
        sat_pos: Satellite ECEF position [m] (3,)
        ground_pos: Ground reference ECEF position [m] (3,)
    
    Returns:
        Elevation angle [degrees]
    """
    # Vector from ground to satellite
    los_vec = sat_pos - ground_pos
    los_norm = np.linalg.norm(los_vec)
    
    if los_norm < 1e-6:
        return 90.0
    
    # Local vertical (away from Earth center)
    ground_norm = np.linalg.norm(ground_pos)
    if ground_norm < 1e-6:
        return 0.0
    
    up_vec = ground_pos / ground_norm
    
    # Elevation = 90° - zenith angle
    cos_zenith = np.dot(los_vec, up_vec) / los_norm
    elevation_rad = np.arcsin(cos_zenith)
    
    return np.degrees(elevation_rad)


def filter_visible_satellites(
    satellites: List[SatelliteObservation],
    reference_pos: np.ndarray,
    min_elevation_deg: float = 15.0
) -> List[SatelliteObservation]:
    """
    Filter satellites by minimum elevation angle.
    
    Args:
        satellites: List of satellite observations
        reference_pos: Reference ground position for elevation check [m]
        min_elevation_deg: Minimum elevation angle [degrees]
    
    Returns:
        List of visible satellites
    """
    visible = []
    
    for sat in satellites:
        elev = compute_elevation_angle(sat.position, reference_pos)
        if elev >= min_elevation_deg:
            visible.append(sat)
    
    return visible


def predict_tdoa(sat_pos: np.ndarray, ref_pos: np.ndarray, emitter_pos: np.ndarray) -> float:
    """
    Predict TDOA given satellite positions and emitter location.
    
    Args:
        sat_pos: Satellite position [m] (3,)
        ref_pos: Reference satellite position [m] (3,)
        emitter_pos: Emitter position [m] (3,)
    
    Returns:
        Predicted TDOA [seconds]
    """
    c = 3e8  # Speed of light
    
    d_sat = np.linalg.norm(emitter_pos - sat_pos)
    d_ref = np.linalg.norm(emitter_pos - ref_pos)
    
    return (d_sat - d_ref) / c


def coarse_position_estimate(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute coarse position estimate using simple WLS.
    
    Args:
        satellites: List of satellite observations (excluding reference)
        ref_sat: Reference satellite
        weights: Optional weights for each observation
    
    Returns:
        Coarse position estimate [m] (3,)
    """
    from scipy.optimize import least_squares
    
    c = 3e8
    
    if weights is None:
        weights = np.ones(len(satellites))
    
    def residual_func(pos):
        residuals = []
        for sat, w in zip(satellites, weights):
            pred_tdoa = predict_tdoa(sat.position, ref_sat.position, pos)
            residuals.append((sat.obs_tdoa - pred_tdoa) * np.sqrt(w))
        return np.array(residuals)
    
    # Initial guess: centroid of satellites
    all_sats = [sat.position for sat in satellites] + [ref_sat.position]
    x0 = np.mean(all_sats, axis=0)
    
    # Bounds
    lo = np.array([-8e5, -8e5, -1e5])
    hi = np.array([8e5, 8e5, 1e6])
    
    result = least_squares(
        residual_func, x0,
        method='trf',
        loss='soft_l1',
        bounds=(lo, hi),
        max_nfev=500
    )
    
    return result.x


def compute_residuals(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    emitter_pos: np.ndarray
) -> Dict[int, float]:
    """
    Compute TDOA residuals for each satellite.
    
    Args:
        satellites: List of satellite observations
        ref_sat: Reference satellite
        emitter_pos: Estimated emitter position [m]
    
    Returns:
        Dictionary mapping satellite ID to residual [seconds]
    """
    residuals = {}
    
    for sat in satellites:
        pred_tdoa = predict_tdoa(sat.position, ref_sat.position, emitter_pos)
        residuals[sat.id] = abs(sat.obs_tdoa - pred_tdoa)
    
    return residuals


def compute_gdop(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    emitter_pos: np.ndarray
) -> float:
    """
    Compute Geometric Dilution of Precision (GDOP).
    
    Args:
        satellites: List of satellite observations
        ref_sat: Reference satellite
        emitter_pos: Emitter position for geometry computation [m]
    
    Returns:
        GDOP value (lower is better)
    """
    H = []
    
    for sat in satellites:
        d_sat = np.linalg.norm(emitter_pos - sat.position)
        d_ref = np.linalg.norm(emitter_pos - ref_sat.position)
        
        if d_sat < 1e-6 or d_ref < 1e-6:
            continue
        
        grad = (emitter_pos - sat.position) / d_sat - (emitter_pos - ref_sat.position) / d_ref
        H.append(grad)
    
    H = np.array(H)
    
    if H.shape[0] < 3:
        return float('inf')
    
    try:
        HTH_inv = np.linalg.inv(H.T @ H)
        gdop = np.sqrt(np.trace(HTH_inv))
        return gdop
    except np.linalg.LinAlgError:
        return float('inf')


def select_satellites_irls(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    max_iters: int = 10,
    residual_threshold_sigma: float = 3.0,
    target_count: int = 12
) -> Tuple[List[SatelliteObservation], np.ndarray]:
    """
    Select best satellites using Iteratively Reweighted Least Squares (IRLS).
    
    Algorithm:
    1. Compute coarse estimate with all satellites
    2. Compute residuals
    3. Compute IRLS weights: w_i = 1 / (ε + residual_i²)
    4. Re-solve with weights
    5. Repeat until convergence
    6. Trim satellites with large residuals
    
    Args:
        satellites: List of visible satellites
        ref_sat: Reference satellite
        max_iters: Maximum IRLS iterations
        residual_threshold_sigma: Threshold for outlier removal (multiples of σ)
        target_count: Target number of satellites to select
    
    Returns:
        (selected_satellites, final_weights)
    """
    if len(satellites) < 3:
        return satellites, np.ones(len(satellites))
    
    # Initial uniform weights
    weights = np.ones(len(satellites))
    eps = 1e-6
    
    # IRLS iterations
    for iteration in range(max_iters):
        # Estimate position with current weights
        pos_est = coarse_position_estimate(satellites, ref_sat, weights)
        
        # Compute residuals
        residuals_dict = compute_residuals(satellites, ref_sat, pos_est)
        residuals = np.array([residuals_dict[sat.id] for sat in satellites])
        
        # Update weights: w_i = 1 / (ε + r_i²)
        weights = 1.0 / (eps + residuals**2)
        weights = weights / (np.sum(weights) + eps)  # Normalize
    
    # Final position estimate
    pos_final = coarse_position_estimate(satellites, ref_sat, weights)
    residuals_dict = compute_residuals(satellites, ref_sat, pos_final)
    residuals = np.array([residuals_dict[sat.id] for sat in satellites])
    
    # Compute threshold for outlier removal
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    sigma_robust = 1.4826 * mad  # Robust std estimate
    threshold = residual_threshold_sigma * sigma_robust
    
    # Select satellites below threshold
    selected = []
    selected_weights = []
    
    for sat, res, w in zip(satellites, residuals, weights):
        if res <= threshold:
            selected.append(sat)
            selected_weights.append(w)
    
    # If too many, keep top-N by weight
    if len(selected) > target_count:
        indices = np.argsort(selected_weights)[::-1][:target_count]
        selected = [selected[i] for i in indices]
        selected_weights = [selected_weights[i] for i in indices]
    
    return selected, np.array(selected_weights)


def select_satellites_gdop(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    coarse_pos: np.ndarray,
    target_count: int = 12
) -> List[SatelliteObservation]:
    """
    Select satellites to minimize GDOP (greedy algorithm).
    
    Args:
        satellites: List of candidate satellites
        ref_sat: Reference satellite
        coarse_pos: Coarse position estimate for geometry computation
        target_count: Number of satellites to select
    
    Returns:
        Selected satellites with best geometry
    """
    if len(satellites) <= target_count:
        return satellites
    
    selected = []
    remaining = list(satellites)
    
    # Greedy selection: add satellite that most reduces GDOP
    for _ in range(min(target_count, len(satellites))):
        best_sat = None
        best_gdop = float('inf')
        
        for sat in remaining:
            trial_selected = selected + [sat]
            gdop = compute_gdop(trial_selected, ref_sat, coarse_pos)
            
            if gdop < best_gdop:
                best_gdop = gdop
                best_sat = sat
        
        if best_sat is not None:
            selected.append(best_sat)
            remaining.remove(best_sat)
    
    return selected


def select_satellites_hybrid(
    satellites: List[SatelliteObservation],
    ref_sat: SatelliteObservation,
    min_elevation_deg: float = 15.0,
    target_count: int = 12,
    use_gdop: bool = True,
    use_irls: bool = True
) -> Tuple[List[SatelliteObservation], Dict]:
    """
    Hybrid satellite selection combining visibility, IRLS, and GDOP optimization.
    
    Algorithm:
    1. Filter by minimum elevation angle
    2. Coarse position estimate with all visible satellites
    3. IRLS to remove outliers and get robust weights
    4. GDOP optimization for best geometry
    5. Final selection of target_count satellites
    
    Args:
        satellites: All available satellites
        ref_sat: Reference satellite
        min_elevation_deg: Minimum elevation angle [degrees]
        target_count: Target number of satellites to select
        use_gdop: Enable GDOP optimization
        use_irls: Enable IRLS outlier removal
    
    Returns:
        (selected_satellites, info_dict)
    """
    info = {
        'total_satellites': len(satellites),
        'visible_satellites': 0,
        'after_irls': 0,
        'final_count': 0,
        'coarse_position': None,
        'gdop': None
    }
    
    # Step 1: Visibility filtering
    # Use centroid of all satellites as reference for elevation
    all_positions = [sat.position for sat in satellites] + [ref_sat.position]
    reference_ground = np.mean(all_positions, axis=0)
    reference_ground[2] = 0  # Project to ground level
    
    visible_sats = filter_visible_satellites(satellites, reference_ground, min_elevation_deg)
    info['visible_satellites'] = len(visible_sats)
    
    if len(visible_sats) < 3:
        print(f"⚠️ Insufficient visible satellites: {len(visible_sats)}")
        return visible_sats, info
    
    # Step 2: Coarse position estimate
    coarse_pos = coarse_position_estimate(visible_sats, ref_sat)
    info['coarse_position'] = coarse_pos
    
    # Step 3: IRLS outlier removal
    selected_sats = visible_sats
    weights = None
    
    if use_irls:
        selected_sats, weights = select_satellites_irls(
            visible_sats, ref_sat, target_count=target_count
        )
        info['after_irls'] = len(selected_sats)
    
    # Step 4: GDOP optimization
    if use_gdop and len(selected_sats) > target_count:
        selected_sats = select_satellites_gdop(
            selected_sats, ref_sat, coarse_pos, target_count
        )
    
    # Final GDOP computation
    if len(selected_sats) >= 3:
        info['gdop'] = compute_gdop(selected_sats, ref_sat, coarse_pos)
    
    info['final_count'] = len(selected_sats)
    
    return selected_sats, info


# Example usage and testing
if __name__ == "__main__":
    print("Satellite Selection Module")
    print("=" * 50)
    
    # Create dummy satellites for testing
    np.random.seed(42)
    
    satellites = []
    for i in range(20):
        # Random LEO orbit positions
        r = 550e3 + 6371e3  # 550 km altitude
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        sat = SatelliteObservation(
            id=i,
            position=np.array([x, y, z]),
            velocity=np.random.randn(3) * 7500,
            rx_signal=np.random.randn(1000),
            obs_tdoa=np.random.uniform(-0.001, 0.001),
            snr=np.random.uniform(10, 30)
        )
        satellites.append(sat)
    
    ref_sat = satellites[0]
    test_sats = satellites[1:]
    
    # Test hybrid selection
    selected, info = select_satellites_hybrid(
        test_sats, ref_sat,
        min_elevation_deg=15.0,
        target_count=12
    )
    
    print(f"\nTotal satellites: {info['total_satellites']}")
    print(f"Visible satellites: {info['visible_satellites']}")
    print(f"After IRLS: {info['after_irls']}")
    print(f"Final selected: {info['final_count']}")
    print(f"GDOP: {info['gdop']:.2f}")
    print(f"Selected IDs: {[s.id for s in selected]}")
