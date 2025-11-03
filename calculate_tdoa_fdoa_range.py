#!/usr/bin/env python3
"""
Calculate correct TDOA_MAX and FDOA_MAX for Starlink LEO constellation
"""

import numpy as np

print("="*70)
print("TDOA/FDOA RANGE CALCULATION FOR STARLINK LEO")
print("="*70)

# ========================================
# STARLINK LEO PARAMETERS
# ========================================
print("\n📡 STARLINK CONSTELLATION PARAMETERS:")
print("-"*70)

# Orbital parameters
altitude_km = 550  # km (typical Starlink altitude)
R_earth_km = 6371  # km (Earth radius)
orbital_radius_km = R_earth_km + altitude_km  # 6921 km

# Calculate orbital velocity
G = 6.674e-11  # m^3 kg^-1 s^-2
M_earth = 5.972e24  # kg
orbital_radius_m = orbital_radius_km * 1e3
orbital_velocity = np.sqrt(G * M_earth / orbital_radius_m)  # m/s
orbital_velocity_km_s = orbital_velocity / 1e3  # km/s

print(f"  Altitude:         {altitude_km} km")
print(f"  Orbital radius:   {orbital_radius_km:.1f} km")
print(f"  Orbital velocity: {orbital_velocity_km_s:.2f} km/s")

# RF parameters (from config)
carrier_freq_hz = 28e9  # 28 GHz
c = 3e8  # speed of light (m/s)

print(f"  Carrier frequency: {carrier_freq_hz/1e9:.1f} GHz")

# Ground observer
ground_observer_height = 0  # m (ground level)

# ========================================
# TDOA CALCULATION
# ========================================
print("\n⏱️  TDOA (Time Difference of Arrival) CALCULATION:")
print("-"*70)

# Maximum distance difference occurs when:
# - One satellite directly overhead (minimum distance = altitude)
# - Another satellite at horizon (maximum distance)

# Minimum distance (satellite overhead)
d_min_km = altitude_km
d_min_m = d_min_km * 1e3

# Maximum distance (satellite at horizon)
# Use geometry: tangent from ground to satellite orbit
# elevation angle = 10° (minimum operational)
min_elevation_deg = 10
min_elevation_rad = np.radians(min_elevation_deg)

# Distance to satellite at minimum elevation
# Using law of cosines in triangle: Earth center - Observer - Satellite
# cos(angle_at_center) = sin(elevation)
angle_at_center = np.pi/2 - min_elevation_rad
d_max_m = np.sqrt(R_earth_km**2 + orbital_radius_km**2 - 
                   2 * R_earth_km * orbital_radius_km * np.cos(angle_at_center)) * 1e3

d_max_km = d_max_m / 1e3

print(f"  Min distance (overhead):     {d_min_km:.1f} km")
print(f"  Max distance (elevation 10°): {d_max_km:.1f} km")

# Maximum TDOA
max_tdoa_s = (d_max_m - d_min_m) / c
max_tdoa_ms = max_tdoa_s * 1e3
max_tdoa_us = max_tdoa_s * 1e6

print(f"\n  📊 TDOA RANGE:")
print(f"    Max TDOA: {max_tdoa_ms:.2f} ms ({max_tdoa_us:.0f} μs)")

# Practical range (considering typical satellite pairs)
# Two satellites at 45° elevation on opposite sides
d_practical = np.sqrt(R_earth_km**2 + orbital_radius_km**2 - 
                      2 * R_earth_km * orbital_radius_km * np.cos(np.radians(45))) * 1e3
practical_tdoa_s = (d_practical - d_min_m) / c
practical_tdoa_ms = practical_tdoa_s * 1e3

print(f"    Practical TDOA (45° elevation): {practical_tdoa_ms:.2f} ms")

# Conservative estimate (safety factor 1.2)
recommended_tdoa_max_ms = max_tdoa_ms * 1.2
recommended_tdoa_max_s = recommended_tdoa_max_ms / 1e3

print(f"\n  ✅ RECOMMENDED TDOA_MAX: {recommended_tdoa_max_ms:.2f} ms ({recommended_tdoa_max_s:.6f} s)")

# ========================================
# FDOA CALCULATION
# ========================================
print("\n📻 FDOA (Frequency Difference of Arrival) CALCULATION:")
print("-"*70)

# Maximum Doppler occurs when satellite moves directly toward/away from observer
# Doppler shift: f_d = (v/c) * f_carrier

max_doppler_hz = (orbital_velocity / c) * carrier_freq_hz
max_doppler_khz = max_doppler_hz / 1e3

print(f"  Orbital velocity:  {orbital_velocity:.0f} m/s ({orbital_velocity_km_s:.2f} km/s)")
print(f"  Carrier frequency: {carrier_freq_hz/1e9:.1f} GHz")

print(f"\n  📊 FDOA RANGE:")
print(f"    Max Doppler (single sat): ±{max_doppler_khz:.1f} kHz")

# FDOA is the DIFFERENCE between two satellites' Doppler
# Worst case: one approaching at max velocity, one receding at max velocity
max_fdoa_hz = 2 * max_doppler_hz
max_fdoa_khz = max_fdoa_hz / 1e3

print(f"    Max FDOA (difference):    ±{max_fdoa_khz:.1f} kHz")

# Practical scenario (most satellite pairs have smaller relative velocity difference)
practical_fdoa_khz = max_fdoa_khz * 0.7  # 70% of theoretical max

print(f"    Practical FDOA:           ±{practical_fdoa_khz:.1f} kHz")

# Conservative estimate
recommended_fdoa_max_khz = max_fdoa_khz * 1.2
recommended_fdoa_max_hz = recommended_fdoa_max_khz * 1e3

print(f"\n  ✅ RECOMMENDED FDOA_MAX: ±{recommended_fdoa_max_khz:.1f} kHz ({recommended_fdoa_max_hz:.0f} Hz)")

# ========================================
# COMPARISON WITH CURRENT VALUES
# ========================================
print("\n" + "="*70)
print("COMPARISON WITH CURRENT IMPLEMENTATION")
print("="*70)

# Current values (from code)
current_tdoa_max_s = 60 * (1/38400)  # 60 * Ts
current_tdoa_max_ms = current_tdoa_max_s * 1e3
current_fdoa_max_hz = 6144.0
current_fdoa_max_khz = current_fdoa_max_hz / 1e3

print(f"\n❌ CURRENT VALUES (from AIS paper):")
print(f"  TDOA_MAX: {current_tdoa_max_ms:.3f} ms ({current_tdoa_max_s:.6f} s)")
print(f"  FDOA_MAX: ±{current_fdoa_max_khz:.2f} kHz ({current_fdoa_max_hz:.0f} Hz)")

print(f"\n✅ CORRECT VALUES (for Starlink LEO):")
print(f"  TDOA_MAX: {recommended_tdoa_max_ms:.2f} ms ({recommended_tdoa_max_s:.6f} s)")
print(f"  FDOA_MAX: ±{recommended_fdoa_max_khz:.1f} kHz ({recommended_fdoa_max_hz:.0f} Hz)")

# Calculate scaling errors
tdoa_scale_error = recommended_tdoa_max_s / current_tdoa_max_s
fdoa_scale_error = recommended_fdoa_max_hz / current_fdoa_max_hz

print(f"\n⚠️  SCALING ERRORS:")
print(f"  TDOA underscaled by: {tdoa_scale_error:.1f}x")
print(f"  FDOA underscaled by: {fdoa_scale_error:.1f}x")

# ========================================
# IMPACT ANALYSIS
# ========================================
print("\n" + "="*70)
print("IMPACT ON STNN PERFORMANCE")
print("="*70)

print(f"\n❌ PROBLEMS WITH CURRENT SCALING:")
print(f"  1. TDOA values clipped at ±{current_tdoa_max_ms:.3f} ms")
print(f"     → But real TDOA can be up to {max_tdoa_ms:.2f} ms")
print(f"     → {(1 - current_tdoa_max_ms/max_tdoa_ms)*100:.1f}% of samples clipped!")

print(f"\n  2. FDOA values clipped at ±{current_fdoa_max_khz:.2f} kHz")
print(f"     → But real FDOA can be up to ±{max_fdoa_khz:.1f} kHz")
print(f"     → {(1 - current_fdoa_max_khz/max_fdoa_khz)*100:.1f}% of samples clipped!")

print(f"\n  3. Model trained on clipped/wrong scale data")
print(f"     → STNN learns incorrect feature space")
print(f"     → Predictions will have systematic bias")

print(f"\n✅ EXPECTED IMPROVEMENT AFTER FIX:")
print(f"  - TDOA error should reduce by ~{tdoa_scale_error:.1f}x")
print(f"  - FDOA error should reduce by ~{fdoa_scale_error:.1f}x")
print(f"  - Localization accuracy improved significantly")

# ========================================
# RECOMMENDED CODE CHANGES
# ========================================
print("\n" + "="*70)
print("RECOMMENDED CODE CHANGES")
print("="*70)

print(f"\nIn model/stnn_localization.py, line 188-189:")
print(f"  CHANGE FROM:")
print(f"    tdoa_max: float = 60 * (1/38400)  # {current_tdoa_max_s:.6f} s")
print(f"    fdoa_max: float = 6144.0  # Hz")

print(f"\n  CHANGE TO:")
print(f"    tdoa_max: float = {recommended_tdoa_max_s:.6f}  # {recommended_tdoa_max_ms:.2f} ms (Starlink LEO)")
print(f"    fdoa_max: float = {recommended_fdoa_max_hz:.1f}  # ±{recommended_fdoa_max_khz:.1f} kHz (Starlink LEO)")

print("\n⚠️  NOTE: After changing these values, you MUST retrain STNN models!")
print("  Old models were trained with wrong normalization scale.")

print("\n" + "="*70)
