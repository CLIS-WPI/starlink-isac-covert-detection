#!/usr/bin/env python3
"""
Quick test to verify ECEF to local coordinate conversion for satellite positions.
"""

import numpy as np
from datetime import datetime, timezone
from core.constellation_select import select_target_and_sensors

def test_ecef_conversion():
    print("="*60)
    print("Testing ECEF to Local Coordinate Conversion")
    print("="*60)
    
    # Select satellites using TLE
    result = select_target_and_sensors(
        tle_path="data/starlink.txt",
        obs_time=datetime.now(timezone.utc),
        target_name_contains="STARLINK",
        target_index=None,
        num_sensors=4,  # Just test with 4 satellites
        min_elevation_deg=15.0,
        check_visibility=True
    )
    
    ground_obs = result['ground_observer']
    print(f"\n✓ Ground Observer (ECEF):")
    print(f"  x={ground_obs[0]/1e6:8.3f} Mm, y={ground_obs[1]/1e6:8.3f} Mm, z={ground_obs[2]/1e6:8.3f} Mm")
    print(f"  Magnitude: {np.linalg.norm(ground_obs)/1e6:.3f} Mm")
    
    print(f"\n✓ Selected Satellites:")
    for i, sensor in enumerate(result['sensors']):
        r_ecef = np.array(sensor['r_ecef'])
        
        # Original ECEF
        print(f"\n  Satellite {i+1}: {sensor['name']}")
        print(f"    ECEF: x={r_ecef[0]/1e6:8.3f} Mm, y={r_ecef[1]/1e6:8.3f} Mm, z={r_ecef[2]/1e6:8.3f} Mm")
        print(f"    ECEF Magnitude: {np.linalg.norm(r_ecef)/1e6:.3f} Mm")
        
        # Convert to local (relative to ground)
        r_local = r_ecef - ground_obs
        print(f"    Local (relative): x={r_local[0]/1e3:8.1f} km, y={r_local[1]/1e3:8.1f} km, z={r_local[2]/1e3:8.1f} km")
        
        # Compute altitude (distance from Earth center - ground distance)
        r_ecef_mag = np.linalg.norm(r_ecef)
        ground_mag = np.linalg.norm(ground_obs)
        altitude = r_ecef_mag - ground_mag
        print(f"    Altitude: {altitude/1e3:.1f} km")
        
        # Final position for simulation
        pos_sim = np.array([r_local[0], r_local[1], altitude])
        print(f"    Simulation coords: x={pos_sim[0]/1e3:8.1f} km, y={pos_sim[1]/1e3:8.1f} km, z={pos_sim[2]/1e3:8.1f} km")
        
        # Sanity check: altitude should be 300-600 km for Starlink
        if 300e3 < altitude < 800e3:
            print(f"    ✓ Altitude in valid range for LEO")
        else:
            print(f"    ⚠️ WARNING: Altitude {altitude/1e3:.1f} km outside expected LEO range")
    
    print("\n" + "="*60)
    print("Conversion test complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_ecef_conversion()
    except FileNotFoundError:
        print("⚠️ TLE file not found. Make sure data/starlink.txt exists.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
