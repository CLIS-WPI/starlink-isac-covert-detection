#!/usr/bin/env python3
"""
Quick test to check ECEF values from constellation_select
"""

from core.constellation_select import select_target_and_sensors
from datetime import datetime, timezone
import numpy as np

print("="*60)
print("ECEF ALTITUDE DEBUG TEST")
print("="*60)

try:
    sel = select_target_and_sensors(
        tle_path='data/starlink.txt',
        obs_time=datetime.now(timezone.utc),
        target_name_contains='STARLINK',
        target_index=None,
        num_sensors=12,
        use_gdop_optimization=True
    )
    
    print(f"\n✓ constellation_select succeeded")
    print(f"  Sensors returned: {len(sel['sensors'])}")
    
    ground_obs = sel['ground_observer']
    print(f"\n📍 Ground Observer:")
    print(f"  Position: {ground_obs}")
    print(f"  |ground_obs| = {np.linalg.norm(ground_obs)/1e3:.1f} km")
    
    print(f"\n🛰️ First 3 satellites:")
    for i, s in enumerate(sel['sensors'][:3]):
        r_ecef = np.array(s['r_ecef'])
        r_ecef_mag = np.linalg.norm(r_ecef)
        R_EARTH = 6371e3
        altitude = r_ecef_mag - R_EARTH
        
        print(f"\n  Satellite {i}:")
        print(f"    r_ecef = [{r_ecef[0]/1e6:.3f}, {r_ecef[1]/1e6:.3f}, {r_ecef[2]/1e6:.3f}] Mm")
        print(f"    |r_ecef| = {r_ecef_mag/1e3:.1f} km")
        print(f"    altitude = {r_ecef_mag/1e3:.1f} - 6371.0 = {altitude/1e3:.1f} km")
        
        if altitude < 0:
            print(f"    ❌ NEGATIVE ALTITUDE!")
        elif altitude < 200e3:
            print(f"    ⚠️ Too low (below LEO)")
        elif altitude > 2000e3:
            print(f"    ⚠️ Too high (above LEO)")
        else:
            print(f"    ✅ Valid LEO altitude")
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    
    # Check if r_ecef values look reasonable
    first_sat = np.array(sel['sensors'][0]['r_ecef'])
    first_mag = np.linalg.norm(first_sat)
    
    if first_mag < 1e6:  # Less than 1000 km
        print("❌ PROBLEM: r_ecef magnitude too small!")
        print("   → Looks like data is in km, not meters")
        print("   → Check leo_orbit.py line 102: r_eci * 1e3")
    elif first_mag < R_EARTH:
        print("❌ PROBLEM: r_ecef < Earth radius!")
        print("   → Satellite inside Earth?!")
    elif (r_ecef_mag - R_EARTH) < 0:
        print("❌ PROBLEM: Calculated altitude is negative")
        print("   → Something wrong with R_EARTH or calculation")
    else:
        print("✅ r_ecef values look reasonable")
    
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
