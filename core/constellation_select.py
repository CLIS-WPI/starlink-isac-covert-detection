# ======================================
# core/constellation_select.py
# Select 1 target Starlink + N sensor Starlinks using TLE + SGP4
# Features:
#  - Real-time position/velocity from SGP4 propagation
#  - Visibility filtering (elevation angle check)
#  - Angular separation ranking for sensor selection
# ======================================

import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from core.leo_orbit import read_tle_file, propagate_tle, SatState

# --- Constants ---
R_EARTH = 6378137.0  # Earth radius [m]
MIN_ELEVATION_DEG = 15.0  # Minimum elevation angle for visibility


def _angular_sep(u: np.ndarray, v: np.ndarray) -> float:
    """Compute angular separation between two 3D vectors."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    cosang = np.clip(np.dot(u, v), -1.0, 1.0)
    return float(np.arccos(cosang))


def compute_elevation_angle(sat_ecef: np.ndarray, ground_ecef: np.ndarray) -> float:
    """
    Compute elevation angle from ground observer to satellite.
    
    Args:
        sat_ecef: Satellite ECEF position [m] (3,)
        ground_ecef: Ground observer ECEF position [m] (3,)
    
    Returns:
        Elevation angle [degrees]
    """
    # Line-of-sight vector from ground to satellite
    los_vec = sat_ecef - ground_ecef
    los_norm = np.linalg.norm(los_vec)
    
    if los_norm < 1e-6:
        return 90.0  # Satellite at observer position
    
    # Local vertical (radial from Earth center)
    ground_norm = np.linalg.norm(ground_ecef)
    if ground_norm < 1e-6:
        return 0.0  # Observer at Earth center (shouldn't happen)
    
    local_vertical = ground_ecef / ground_norm
    
    # Elevation = 90° - zenith_angle
    cos_zenith = np.dot(los_vec, local_vertical) / los_norm
    elevation_rad = np.arcsin(np.clip(cos_zenith, -1.0, 1.0))
    
    return np.degrees(elevation_rad)


def estimate_ground_observer(target_ecef: np.ndarray, altitude_m: float = 0.0) -> np.ndarray:
    """
    Estimate ground observer position below satellite (nadir point).
    
    Args:
        target_ecef: Target satellite ECEF position [m]
        altitude_m: Observer altitude above sea level [m]
    
    Returns:
        Ground observer ECEF position [m] (3,)
    """
    # Project satellite to ground (nadir point)
    r_target = np.linalg.norm(target_ecef)
    if r_target < 1e-6:
        return np.array([R_EARTH + altitude_m, 0.0, 0.0])
    
    # Ground position: same direction but at R_EARTH + altitude
    ground_ecef = target_ecef * (R_EARTH + altitude_m) / r_target
    
    return ground_ecef

def select_target_and_sensors(
    tle_path: str,
    obs_time: Optional[datetime] = None,
    target_name_contains: Optional[str] = "STARLINK",
    target_index: Optional[int] = None,
    num_sensors: int = 12,
    ground_observer: Optional[np.ndarray] = None,
    min_elevation_deg: float = MIN_ELEVATION_DEG,
    check_visibility: bool = True,
    use_gdop_optimization: bool = False  # 🔧 NEW: Enable GDOP-based selection
) -> Dict:
    """
    Select target satellite and sensor satellites from TLE constellation.
    
    Args:
        tle_path: Path to TLE file
        obs_time: Observation time (UTC), defaults to now
        target_name_contains: Filter TLEs by name pattern (e.g., "STARLINK")
        target_index: Index of target satellite (None = first/random)
        num_sensors: Number of sensor satellites to select
        ground_observer: Ground observer ECEF position [m] (3,)
                        If None, computed as nadir point below target
        min_elevation_deg: Minimum elevation angle for visibility [degrees]
        check_visibility: Enable visibility filtering based on elevation
    
    Returns:
        Dictionary with:
          {
            'target': {'name', 'r_ecef', 'v_ecef'},
            'sensors': [{'name','r_ecef','v_ecef'}, ... num_sensors],
            'ground_observer': ground_ecef (3,),
            'elevations': [elevation angles for sensors]
          }
    """
    if obs_time is None:
        obs_time = datetime.now(timezone.utc)

    # Read and filter TLEs
    tles = read_tle_file(tle_path)
    if not tles:
        raise ValueError("No TLEs found in file")

    if target_name_contains:
        tles_filtered = [t for t in tles if target_name_contains.upper() in t.name.upper()]
    else:
        tles_filtered = tles[:]

    if not tles_filtered:
        raise ValueError(f"No TLEs matched pattern: {target_name_contains}")

    # Choose target satellite
    if target_index is not None:
        target_tle = tles_filtered[target_index % len(tles_filtered)]
    else:
        target_tle = tles_filtered[0]  # Or random: np.random.choice(tles_filtered)

    # Propagate all satellites to obs_time
    states: List[SatState] = []
    for tle in tles_filtered:
        try:
            st = propagate_tle(tle, obs_time)
            states.append(st)
        except Exception as e:
            # Skip satellites with propagation errors
            continue

    if len(states) < num_sensors + 1:
        raise ValueError(f"Insufficient satellites: {len(states)} < {num_sensors + 1}")

    # Find target state
    target_state = None
    for st in states:
        if st.name == target_tle.name:
            target_state = st
            break
    
    if target_state is None:
        # Fallback: use first state as target
        target_state = states[0]

    # Determine ground observer position
    if ground_observer is None:
        # Use nadir point below target satellite
        ground_observer = estimate_ground_observer(target_state.r_ecef_m, altitude_m=0.0)
    
    # Select sensor satellites
    others = [s for s in states if s.name != target_state.name]
    
    if check_visibility:
        # Filter by elevation angle (visibility check)
        visible_sensors = []
        for s in others:
            elev = compute_elevation_angle(s.r_ecef_m, ground_observer)
            if elev >= min_elevation_deg:
                visible_sensors.append((s, elev))
        
        if len(visible_sensors) < num_sensors:
            print(f"⚠️ Warning: Only {len(visible_sensors)} visible satellites (min elev={min_elevation_deg}°)")
            print(f"   Requested {num_sensors} sensors. Using all visible satellites.")
        
        # Rank by angular separation from target
        angs = [_angular_sep(target_state.r_ecef_m, s.r_ecef_m) for s, _ in visible_sensors]
        
        if use_gdop_optimization and len(visible_sensors) > num_sensors:
            # For GDOP optimization: start with DIVERSE satellites (max angular separation)
            order = np.argsort(angs)[::-1]  # 🔧 FIXED: Furthest satellites for better geometry
        else:
            # Without GDOP: use closest satellites (original behavior)
            order = np.argsort(angs)  # Closest satellites in orbit
        
        selected = [visible_sensors[i] for i in order[:num_sensors]]
        sensors = [s for s, _ in selected]
        elevations = [elev for _, elev in selected]
        
        # 🔧 NEW: GDOP-based optimization (greedy selection)
        if use_gdop_optimization and len(visible_sensors) > num_sensors:
            try:
                from core.localization import compute_gdop_tdoa
                
                # Start with 4 satellites with maximum angular diversity
                # Pick satellites roughly 90° apart for tetrahedral geometry
                n_candidates = len(sensors)
                if n_candidates >= 4:
                    indices = [0, n_candidates//4, n_candidates//2, 3*n_candidates//4]
                    best_sensors = [sensors[i] for i in indices[:4]]
                else:
                    best_sensors = sensors[:4]
                
                best_gdop = float('inf')
                
                # Estimate emitter position (use ground observer as initial guess)
                emitter_guess = np.array([ground_observer[0], ground_observer[1], 0.0])
                
                # Greedy: add satellites one by one to minimize GDOP
                remaining = [s for s in sensors if s not in best_sensors]
                for _ in range(num_sensors - 4):
                    if not remaining:
                        break
                    
                    best_addition = None
                    min_gdop = best_gdop
                    
                    for candidate in remaining:
                        test_sats = best_sensors + [candidate]
                        test_positions = [s.r_ecef_m for s in test_sats]
                        
                        try:
                            gdop = compute_gdop_tdoa(test_positions, emitter_guess)
                            if gdop < min_gdop:
                                min_gdop = gdop
                                best_addition = candidate
                        except:
                            continue
                    
                    if best_addition:
                        best_sensors.append(best_addition)
                        remaining.remove(best_addition)
                        best_gdop = min_gdop
                
                sensors = best_sensors
                print(f"✓ GDOP optimization: {len(sensors)} satellites, GDOP={best_gdop:.2f}")
                
            except ImportError:
                print("⚠️ GDOP optimization failed: compute_gdop_tdoa not available")
        
    else:
        # No visibility check: rank all by angular separation
        angs = [_angular_sep(target_state.r_ecef_m, s.r_ecef_m) for s in others]
        order = np.argsort(angs)
        sensors = [others[i] for i in order[:num_sensors]]
        
        # Compute elevations for info
        elevations = [compute_elevation_angle(s.r_ecef_m, ground_observer) for s in sensors]

    return {
        "target": {
            "name": target_state.name,
            "r_ecef": target_state.r_ecef_m,
            "v_ecef": target_state.v_ecef_mps
        },
        "sensors": [
            {"name": s.name, "r_ecef": s.r_ecef_m, "v_ecef": s.v_ecef_mps}
            for s in sensors
        ],
        "ground_observer": ground_observer,
        "elevations": elevations
    }


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Constellation Selection with Visibility Check")
    print("="*60)
    
    # Test with dummy TLE file path
    tle_path = "data/starlink.txt"
    
    try:
        result = select_target_and_sensors(
            tle_path=tle_path,
            num_sensors=12,
            min_elevation_deg=15.0,
            check_visibility=True
        )
        
        print(f"\n✓ Target: {result['target']['name']}")
        print(f"  Position: [{result['target']['r_ecef'][0]/1e6:.2f}, "
              f"{result['target']['r_ecef'][1]/1e6:.2f}, "
              f"{result['target']['r_ecef'][2]/1e6:.2f}] Mm")
        
        print(f"\n✓ Ground Observer:")
        print(f"  Position: [{result['ground_observer'][0]/1e6:.2f}, "
              f"{result['ground_observer'][1]/1e6:.2f}, "
              f"{result['ground_observer'][2]/1e6:.2f}] Mm")
        
        print(f"\n✓ Selected {len(result['sensors'])} sensor satellites:")
        for i, (sensor, elev) in enumerate(zip(result['sensors'], result['elevations'])):
            print(f"  [{i+1:2d}] {sensor['name']:<30s} Elevation: {elev:5.1f}°")
        
        print(f"\n✓ Elevation Statistics:")
        elevations = result['elevations']
        print(f"  Min:    {min(elevations):5.1f}°")
        print(f"  Max:    {max(elevations):5.1f}°")
        print(f"  Mean:   {np.mean(elevations):5.1f}°")
        print(f"  Median: {np.median(elevations):5.1f}°")
        
    except FileNotFoundError:
        print(f"\n⚠️ TLE file not found: {tle_path}")
        print("   This is expected if running outside the main project directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
