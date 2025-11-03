#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DATASET VALIDATOR
Checks for ALL potential issues in the generated dataset
"""

import pickle
import numpy as np
import sys
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

# ============================================================================
# CONFIGURATION
# ============================================================================
dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"

# Expected ranges for validation
EXPECTED_RANGES = {
    'satellite_altitude_km': (400, 650),     # LEO Starlink altitude
    'satellite_velocity_mps': (7000, 8000),  # Orbital velocity
    'emitter_altitude_m': (-100, 100),       # Ground level ±100m
    'ground_observer_altitude_km': (6300, 6500),  # Earth radius range
    'power_ratio': (0.7, 1.5),               # Attack/Benign power (relaxed for channel variance)
    'snr_db': (10, 30),                      # Signal quality
    'tdoa_range_ms': (-10, 10),              # TDOA range (for LEO)
    'fdoa_range_khz': (-100, 100),           # FDOA range (for LEO)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_subheader(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}")

def check_status(passed, message):
    """Print status with emoji"""
    if passed:
        print(f"✅ {message}")
        return True
    else:
        print(f"❌ {message}")
        return False

def warn_status(message):
    """Print warning"""
    print(f"⚠️  {message}")

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

def check_basic_structure(dataset):
    """Check if dataset has all required keys"""
    print_subheader("1. BASIC STRUCTURE")
    
    required_keys = [
        'iq_samples', 'csi', 'radar_echo', 'labels', 
        'emitter_locations', 'satellite_receptions', 
        'sampling_rate', 'tx_time_padded', 'rx_time_b_full'
    ]
    
    all_passed = True
    for key in required_keys:
        passed = key in dataset
        all_passed &= check_status(passed, f"Key '{key}' exists")
    
    return all_passed


def check_sample_counts(dataset):
    """Check if sample counts are consistent"""
    print_subheader("2. SAMPLE COUNTS")
    
    labels = dataset['labels']
    n_total = len(labels)
    n_benign = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    
    print(f"  Total samples:  {n_total}")
    print(f"  Benign:         {n_benign}")
    print(f"  Attack:         {n_attack}")
    
    all_passed = True
    all_passed &= check_status(n_benign == n_attack, 
                                f"Balanced classes (50/50 split)")
    all_passed &= check_status(n_total >= 100, 
                                f"Sufficient samples (≥100)")
    
    # Check emitter locations
    n_emitter = sum(1 for loc in dataset['emitter_locations'] if loc is not None)
    all_passed &= check_status(n_emitter == n_attack, 
                                f"All attack samples have emitter locations ({n_emitter}/{n_attack})")
    
    return all_passed


def check_nan_inf(dataset):
    """Check for NaN/Inf in all numeric arrays"""
    print_subheader("3. NaN/Inf CHECK")
    
    all_passed = True
    
    # Check IQ samples
    iq = dataset['iq_samples']
    has_nan = np.any(np.isnan(np.abs(iq)))
    has_inf = np.any(np.isinf(np.abs(iq)))
    all_passed &= check_status(not has_nan and not has_inf, 
                                f"IQ samples clean (no NaN/Inf)")
    
    # Check CSI
    csi = dataset['csi']
    has_nan = np.any(np.isnan(np.abs(csi)))
    has_inf = np.any(np.isinf(np.abs(csi)))
    all_passed &= check_status(not has_nan and not has_inf, 
                                f"CSI clean (no NaN/Inf)")
    
    # Check satellite receptions
    nan_count = 0
    for sat_list in dataset['satellite_receptions']:
        for sat in sat_list:
            if np.any(np.isnan(np.abs(sat['rx_time']))):
                nan_count += 1
    all_passed &= check_status(nan_count == 0, 
                                f"Satellite receptions clean ({nan_count} NaN found)")
    
    return all_passed


def check_power_ratio(dataset):
    """Check if attack signals have covert power (≈1.0)"""
    print_subheader("4. POWER ANALYSIS")
    
    labels = dataset['labels']
    iq_samples = dataset['iq_samples']
    
    # Calculate powers
    benign_powers = [np.mean(np.abs(iq_samples[i])**2) 
                     for i in range(len(labels)) if labels[i] == 0]
    attack_powers = [np.mean(np.abs(iq_samples[i])**2) 
                     for i in range(len(labels)) if labels[i] == 1]
    
    benign_mean = np.mean(benign_powers)
    benign_std = np.std(benign_powers)
    attack_mean = np.mean(attack_powers)
    attack_std = np.std(attack_powers)
    ratio = attack_mean / benign_mean if benign_mean > 0 else 0
    
    # Coefficient of variation (CV = std/mean)
    benign_cv = benign_std / benign_mean if benign_mean > 0 else 0
    attack_cv = attack_std / attack_mean if attack_mean > 0 else 0
    
    print(f"  Benign power (mean): {benign_mean:.6e}")
    print(f"  Benign CV:           {benign_cv:.4f}")
    print(f"  Attack power (mean): {attack_mean:.6e}")
    print(f"  Attack CV:           {attack_cv:.4f}")
    print(f"  Power ratio:         {ratio:.4f}")
    
    all_passed = True
    
    # Check 1: Power ratio (relaxed for natural channel variance)
    min_ratio, max_ratio = EXPECTED_RANGES['power_ratio']
    if min_ratio <= ratio <= max_ratio:
        all_passed &= check_status(True, 
                                    f"Power ratio in range ({min_ratio:.2f} - {max_ratio:.2f})")
    else:
        # Warning only, not fatal (due to channel variance)
        warn_status(f"Power ratio {ratio:.2f} outside expected range (may be OK with channel variance)")
    
    # Check 2: Variance preservation (CRITICAL for covert detection!)
    if attack_cv > 0.5:
        all_passed &= check_status(True, 
                                    f"Natural variance preserved (attack CV={attack_cv:.2f})")
    elif attack_cv > 0.1:
        warn_status(f"Moderate variance (attack CV={attack_cv:.2f})")
    else:
        all_passed &= check_status(False, 
                                    f"❌ No variance! All attacks identical (CV={attack_cv:.4f})")
    
    # Check 3: Similar variance between benign and attack
    cv_ratio = attack_cv / benign_cv if benign_cv > 0 else 0
    print(f"  CV ratio (A/B):      {cv_ratio:.4f}")
    
    if 0.5 <= cv_ratio <= 2.0:
        all_passed &= check_status(True, 
                                    f"Similar variance structure (CV ratio={cv_ratio:.2f})")
    else:
        warn_status(f"Different variance structure (CV ratio={cv_ratio:.2f})")
    
    return all_passed


def check_emitter_locations(dataset):
    """🔥 CRITICAL: Check if emitters are on GROUND (not in space!)"""
    print_subheader("5. EMITTER LOCATIONS (CRITICAL CHECK)")
    
    labels = dataset['labels']
    emitter_locs = dataset['emitter_locations']
    
    all_passed = True
    issues = []
    
    for i, (label, loc) in enumerate(zip(labels, emitter_locs)):
        if label == 1 and loc is not None:  # Attack sample
            altitude = loc[2]  # z-coordinate
            
            # 🚨 CRITICAL: Emitter must be on ground!
            min_alt, max_alt = EXPECTED_RANGES['emitter_altitude_m']
            if not (min_alt <= altitude <= max_alt):
                issues.append((i, altitude))
    
    if len(issues) == 0:
        all_passed &= check_status(True, 
                                    f"All emitters on ground (altitude ≈ 0 m)")
    else:
        all_passed &= check_status(False, 
                                    f"⚠️ {len(issues)} emitters NOT on ground!")
        print(f"\n  🔴 CRITICAL ERROR: Emitters in space!")
        for i, alt in issues[:5]:  # Show first 5
            print(f"     Sample {i}: altitude = {alt/1e3:.1f} km (should be ≈0)")
        if len(issues) > 5:
            print(f"     ... and {len(issues)-5} more")
    
    return all_passed


def check_satellite_geometry(dataset):
    """Check satellite positions and velocities"""
    print_subheader("6. SATELLITE GEOMETRY")
    
    all_passed = True
    sat_receptions = dataset['satellite_receptions']
    
    if len(sat_receptions) == 0:
        return check_status(False, "No satellite data found")
    
    # Check first sample
    sample_sats = sat_receptions[0]
    
    # Check altitudes
    altitudes = [sat['position'][2] for sat in sample_sats]
    min_alt, max_alt = EXPECTED_RANGES['satellite_altitude_km']
    min_alt_m, max_alt_m = min_alt * 1e3, max_alt * 1e3
    
    bad_altitudes = [alt for alt in altitudes if not (min_alt_m <= alt <= max_alt_m)]
    
    print(f"  Satellites:      {len(sample_sats)}")
    print(f"  Altitude range:  {min(altitudes)/1e3:.1f} - {max(altitudes)/1e3:.1f} km")
    
    all_passed &= check_status(len(bad_altitudes) == 0, 
                                f"Satellite altitudes in LEO range ({min_alt}-{max_alt} km)")
    
    # Check velocities
    velocities = [np.linalg.norm(sat['velocity']) for sat in sample_sats]
    min_vel, max_vel = EXPECTED_RANGES['satellite_velocity_mps']
    
    bad_velocities = [vel for vel in velocities if not (min_vel <= vel <= max_vel)]
    
    print(f"  Velocity range:  {min(velocities)/1e3:.2f} - {max(velocities)/1e3:.2f} km/s")
    
    all_passed &= check_status(len(bad_velocities) == 0, 
                                f"Orbital velocities realistic ({min_vel/1e3:.1f}-{max_vel/1e3:.1f} km/s)")
    
    # Check SNR quality
    snrs = [sat.get('ebno_db', 0) for sat in sample_sats if 'ebno_db' in sat]
    if len(snrs) > 0:
        min_snr, max_snr = EXPECTED_RANGES['snr_db']
        low_snr = [s for s in snrs if s < min_snr]
        
        print(f"  SNR range:       {min(snrs):.1f} - {max(snrs):.1f} dB")
        all_passed &= check_status(len(low_snr) == 0, 
                                    f"SNR sufficient for TDOA/FDOA (≥{min_snr} dB)")
    
    return all_passed


def check_spectral_separability(dataset):
    """Check if attack signals have spectral separability (CRITICAL for detection!)"""
    print_subheader("7. SPECTRAL SEPARABILITY")
    
    labels = dataset['labels']
    iq_samples = dataset['iq_samples']
    
    def compute_spectral_features(signal, n_bins=64):
        """Compute power spectrum features"""
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft)**2
        
        # Divide into frequency bins
        bin_size = len(power_spectrum) // n_bins
        bin_powers = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size
            if end <= len(power_spectrum):
                bin_powers.append(np.mean(power_spectrum[start:end]))
        
        return np.array(bin_powers)
    
    # Sample for speed (200 samples per class)
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    attack_indices = [i for i, label in enumerate(labels) if label == 1]
    
    sample_size = min(200, len(benign_indices), len(attack_indices))
    benign_sample = np.random.choice(benign_indices, sample_size, replace=False)
    attack_sample = np.random.choice(attack_indices, sample_size, replace=False)
    
    print(f"  Analyzing {sample_size} samples per class...")
    
    # Compute spectral features
    benign_spectral = np.array([compute_spectral_features(iq_samples[i]) for i in benign_sample])
    attack_spectral = np.array([compute_spectral_features(iq_samples[i]) for i in attack_sample])
    
    # Compute Cohen's d per frequency bin
    cohens_d_per_bin = []
    for bin_idx in range(benign_spectral.shape[1]):
        benign_bin = benign_spectral[:, bin_idx]
        attack_bin = attack_spectral[:, bin_idx]
        
        benign_mean = np.mean(benign_bin)
        attack_mean = np.mean(attack_bin)
        benign_std = np.std(benign_bin)
        attack_std = np.std(attack_bin)
        
        pooled_std = np.sqrt((benign_std**2 + attack_std**2) / 2)
        if pooled_std > 0:
            d = abs((attack_mean - benign_mean) / pooled_std)
            cohens_d_per_bin.append(d)
    
    mean_spectral_d = np.mean(cohens_d_per_bin)
    max_spectral_d = np.max(cohens_d_per_bin)
    top5_bins = np.argsort(cohens_d_per_bin)[-5:]
    
    print(f"  Mean Cohen's d:      {mean_spectral_d:.4f}")
    print(f"  Max Cohen's d:       {max_spectral_d:.4f}")
    print(f"  Top 5 bins:          {top5_bins.tolist()}")
    print(f"  Top 5 d values:      {[f'{cohens_d_per_bin[i]:.3f}' for i in top5_bins]}")
    
    all_passed = True
    if mean_spectral_d >= 0.2:
        all_passed &= check_status(True, 
                                    f"EXCELLENT separability (d={mean_spectral_d:.3f} ≥ 0.2)")
    elif mean_spectral_d >= 0.1:
        warn_status(f"MODERATE separability (d={mean_spectral_d:.3f})")
        all_passed &= check_status(True, 
                                    f"Acceptable for training (d={mean_spectral_d:.3f} ≥ 0.1)")
    else:
        all_passed &= check_status(False, 
                                    f"POOR separability (d={mean_spectral_d:.3f} < 0.1)")
    
    return all_passed


def check_angular_diversity(dataset):
    """Check if satellites have good angular diversity"""
    print_subheader("8. ANGULAR DIVERSITY")
    
    labels = dataset['labels']
    sat_receptions = dataset['satellite_receptions']
    emitter_locs = dataset['emitter_locations']
    
    # Check multiple attack samples for better statistics
    attack_idx = np.where(labels == 1)[0]
    if len(attack_idx) == 0:
        return check_status(False, "No attack samples to check")
    
    # Sample up to 10 attack samples
    sample_size = min(10, len(attack_idx))
    sample_indices = np.random.choice(attack_idx, sample_size, replace=False)
    
    all_min_angles = []
    all_max_angles = []
    
    for idx in sample_indices:
        sats = sat_receptions[idx]
        emitter = emitter_locs[idx]
        
        if emitter is None or len(sats) < 2:
            continue
        
        # Compute line-of-sight vectors
        los_vectors = []
        for sat in sats:
            vec = np.array(sat['position']) - np.array(emitter)
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-6:
                los_vectors.append(vec / vec_norm)
        
        if len(los_vectors) < 2:
            continue
        
        # Check pairwise angles
        min_angle = float('inf')
        max_angle = 0
        for i in range(len(los_vectors)):
            for j in range(i + 1, len(los_vectors)):
                cos_angle = np.clip(np.dot(los_vectors[i], los_vectors[j]), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                min_angle = min(min_angle, angle_deg)
                max_angle = max(max_angle, angle_deg)
        
        all_min_angles.append(min_angle)
        all_max_angles.append(max_angle)
    
    if len(all_min_angles) == 0:
        return check_status(False, "Could not compute angular diversity")
    
    avg_min_angle = np.mean(all_min_angles)
    avg_max_angle = np.mean(all_max_angles)
    
    print(f"  Samples checked:     {len(all_min_angles)}")
    print(f"  Avg minimum angle:   {avg_min_angle:.1f}°")
    print(f"  Avg maximum angle:   {avg_max_angle:.1f}°")
    
    all_passed = True
    all_passed &= check_status(avg_min_angle >= 10.0, 
                                f"Minimum angular separation ≥10° (good geometry)")
    all_passed &= check_status(avg_max_angle >= 50.0, 
                                f"Maximum angular span ≥50° (diverse coverage)")
    
    return all_passed


def check_tdoa_fdoa_ranges(dataset):
    """Check if TDOA/FDOA values are in expected ranges"""
    print_subheader("9. TDOA/FDOA RANGES")
    
    sat_receptions = dataset['satellite_receptions']
    c = 3e8  # Speed of light
    
    # 🔧 FIX: Get sampling rate from dataset, not hardcoded!
    sampling_rate = dataset.get('sampling_rate', 38.4e6)  # Default: 38.4 MHz
    print(f"  Sampling rate:   {sampling_rate/1e6:.1f} MHz")
    
    all_passed = True
    
    # Sample TDOA values from delays (random sample for speed)
    sample_size = min(100, len(sat_receptions))
    sample_indices = np.random.choice(len(sat_receptions), sample_size, replace=False)
    
    tdoa_values = []
    for idx in sample_indices:
        sat_list = sat_receptions[idx]
        if len(sat_list) < 2:
            continue
        ref_delay = sat_list[0]['true_delay_samples']
        for sat in sat_list[1:]:
            tdoa_samples = sat['true_delay_samples'] - ref_delay
            # Convert to seconds using actual sampling rate
            tdoa_s = tdoa_samples / sampling_rate
            tdoa_values.append(tdoa_s * 1000)  # Convert to ms
    
    if len(tdoa_values) > 0:
        min_tdoa, max_tdoa = EXPECTED_RANGES['tdoa_range_ms']
        print(f"  TDOA range:      {min(tdoa_values):.2f} - {max(tdoa_values):.2f} ms")
        print(f"  TDOA mean:       {np.mean(tdoa_values):.2f} ms")
        
        out_of_range = [t for t in tdoa_values if not (min_tdoa <= t <= max_tdoa)]
        if len(out_of_range) > 0:
            warn_status(f"{len(out_of_range)}/{len(tdoa_values)} TDOA values out of range")
        all_passed &= check_status(len(out_of_range) < len(tdoa_values) * 0.1, 
                                    f"TDOA values mostly in range ({min_tdoa} - {max_tdoa} ms)")
    else:
        warn_status("Could not compute TDOA values")
    
    # Note: FDOA requires velocity info which may not be directly accessible
    # We check velocity magnitudes instead (already done in satellite_geometry)
    
    return all_passed


def run_full_validation(dataset_path):
    """Run all validation checks"""
    print_header("🔍 COMPREHENSIVE DATASET VALIDATION")
    print(f"\nDataset: {dataset_path}")
    
    # Load dataset
    try:
        print("\n📂 Loading dataset...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Calculate memory usage
        try:
            import os
            file_size_mb = os.path.getsize(dataset_path) / (1024**2)
            print(f"✅ Dataset loaded successfully ({file_size_mb:.1f} MB)")
        except:
            print("✅ Dataset loaded successfully")
        
        # Print basic info
        n_samples = len(dataset.get('labels', []))
        sampling_rate = dataset.get('sampling_rate', 0)
        signal_len = len(dataset['iq_samples'][0]) if len(dataset['iq_samples']) > 0 else 0
        
        print(f"\n📊 DATASET INFO:")
        print(f"  Samples:          {n_samples}")
        print(f"  Sampling rate:    {sampling_rate/1e6:.1f} MHz" if sampling_rate > 0 else "  Sampling rate:    Unknown")
        print(f"  Signal length:    {signal_len} samples")
        print(f"  Duration:         {signal_len/sampling_rate*1000:.2f} ms" if sampling_rate > 0 and signal_len > 0 else "  Duration:         Unknown")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Run all checks
    results = {}
    results['structure'] = check_basic_structure(dataset)
    results['counts'] = check_sample_counts(dataset)
    results['nan_inf'] = check_nan_inf(dataset)
    results['power'] = check_power_ratio(dataset)
    results['emitter'] = check_emitter_locations(dataset)  # 🔥 CRITICAL
    results['satellites'] = check_satellite_geometry(dataset)
    results['spectral'] = check_spectral_separability(dataset)  # 🌟 NEW!
    results['diversity'] = check_angular_diversity(dataset)
    results['ranges'] = check_tdoa_fdoa_ranges(dataset)
    
    # Final summary
    print_header("📋 VALIDATION SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"\n  Total checks:  {total_checks}")
    print(f"  Passed:        {passed_checks} ✅")
    print(f"  Failed:        {total_checks - passed_checks} ❌")
    print(f"  Success rate:  {100*passed_checks/total_checks:.1f}%")
    
    if passed_checks == total_checks:
        print("\n  🎉 ALL CHECKS PASSED! Dataset is ready for training.")
    else:
        print("\n  ⚠️  SOME CHECKS FAILED! Review errors above before training.")
        print("\n  Failed checks:")
        for name, passed in results.items():
            if not passed:
                print(f"     ❌ {name}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    run_full_validation(dataset_path)