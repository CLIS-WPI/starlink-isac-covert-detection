#!/usr/bin/env python3
"""
Final Dataset Quality Analysis
Verify all fixes are working correctly
"""

import pickle
import numpy as np
from scipy import stats
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"

print("="*70)
print("FINAL DATASET QUALITY ANALYSIS")
print("="*70)

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

labels = dataset['labels']
iq_samples = dataset['iq_samples']
emitter_locations = dataset['emitter_locations']

# Separate benign and attack
benign_indices = [i for i, label in enumerate(labels) if label == 0]
attack_indices = [i for i, label in enumerate(labels) if label == 1]

print(f"\n📊 DATASET SIZE:")
print(f"  Total samples: {len(labels)}")
print(f"  Benign: {len(benign_indices)}")
print(f"  Attack: {len(attack_indices)}")

# ========== 1. EMITTER ALTITUDE ==========
print(f"\n{'='*70}")
print("1️⃣ EMITTER ALTITUDE")
print("="*70)

attack_altitudes = []
for i in attack_indices:
    emitter = emitter_locations[i]
    if emitter is not None:
        attack_altitudes.append(emitter[2])

if len(attack_altitudes) > 0:
    mean_alt = np.mean(attack_altitudes) / 1e3
    max_alt = np.max(np.abs(attack_altitudes)) / 1e3
    print(f"  Mean altitude: {mean_alt:.3f} km")
    print(f"  Max |altitude|: {max_alt:.3f} km")
    print(f"  ✅ ALL EMITTERS ON GROUND!" if max_alt < 1.0 else f"  ❌ FAIL: Emitters at {max_alt:.1f} km")
else:
    print("  ❌ No emitter locations found!")

# ========== 2. POWER VARIANCE ==========
print(f"\n{'='*70}")
print("2️⃣ POWER VARIANCE (Natural Channel Variance)")
print("="*70)

benign_powers = [np.mean(np.abs(iq_samples[i])**2) for i in benign_indices]
attack_powers = [np.mean(np.abs(iq_samples[i])**2) for i in attack_indices]

benign_mean = np.mean(benign_powers)
benign_std = np.std(benign_powers)
attack_mean = np.mean(attack_powers)
attack_std = np.std(attack_powers)

benign_cv = benign_std / benign_mean if benign_mean > 0 else 0
attack_cv = attack_std / attack_mean if attack_mean > 0 else 0

print(f"  Benign: {benign_mean:.6e} ± {benign_std:.6e} (CV={benign_cv:.4f})")
print(f"  Attack: {attack_mean:.6e} ± {attack_std:.6e} (CV={attack_cv:.4f})")
print(f"  Power ratio: {attack_mean/benign_mean:.4f}")

# Cohen's d for power
if benign_std > 0 and attack_std > 0:
    pooled_std = np.sqrt((benign_std**2 + attack_std**2) / 2)
    cohens_d_power = (attack_mean - benign_mean) / pooled_std if pooled_std > 0 else 0
    print(f"  Cohen's d (power): {cohens_d_power:.4f}")
    
    if attack_cv > 0.5:
        print(f"  ✅ NATURAL VARIANCE PRESERVED (CV={attack_cv:.4f})")
    else:
        print(f"  ⚠️  Low variance (CV={attack_cv:.4f})")
else:
    print(f"  ❌ Zero variance detected!")

# ========== 3. SPECTRAL FEATURES ==========
print(f"\n{'='*70}")
print("3️⃣ SPECTRAL SEPARABILITY (FFT Power Spectrum)")
print("="*70)

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
        bin_powers.append(np.mean(power_spectrum[start:end]))
    
    return np.array(bin_powers)

# Sample 200 signals for speed
sample_size = min(200, len(benign_indices), len(attack_indices))
benign_sample = np.random.choice(benign_indices, sample_size, replace=False)
attack_sample = np.random.choice(attack_indices, sample_size, replace=False)

benign_spectral = [compute_spectral_features(iq_samples[i]) for i in benign_sample]
attack_spectral = [compute_spectral_features(iq_samples[i]) for i in attack_sample]

benign_spectral = np.array(benign_spectral)
attack_spectral = np.array(attack_spectral)

# Compute Cohen's d per frequency bin
cohens_d_per_bin = []
for bin_idx in range(benign_spectral.shape[1]):
    benign_bin = benign_spectral[:, bin_idx]
    attack_bin = attack_spectral[:, bin_idx]
    
    benign_mean_bin = np.mean(benign_bin)
    attack_mean_bin = np.mean(attack_bin)
    benign_std_bin = np.std(benign_bin)
    attack_std_bin = np.std(attack_bin)
    
    pooled_std_bin = np.sqrt((benign_std_bin**2 + attack_std_bin**2) / 2)
    if pooled_std_bin > 0:
        d = (attack_mean_bin - benign_mean_bin) / pooled_std_bin
        cohens_d_per_bin.append(abs(d))
    else:
        cohens_d_per_bin.append(0)

mean_spectral_d = np.mean(cohens_d_per_bin)
max_spectral_d = np.max(cohens_d_per_bin)
top5_bins = np.argsort(cohens_d_per_bin)[-5:]

print(f"  Mean Cohen's d (spectral): {mean_spectral_d:.4f}")
print(f"  Max Cohen's d (spectral): {max_spectral_d:.4f}")
print(f"  Top 5 discriminative bins: {top5_bins.tolist()}")
print(f"  Top 5 Cohen's d values: {[f'{cohens_d_per_bin[i]:.4f}' for i in top5_bins]}")

if mean_spectral_d > 0.2:
    print(f"  ✅ GOOD SPECTRAL SEPARABILITY (d={mean_spectral_d:.4f})")
elif mean_spectral_d > 0.1:
    print(f"  ⚠️  MODERATE SEPARABILITY (d={mean_spectral_d:.4f})")
else:
    print(f"  ❌ POOR SEPARABILITY (d={mean_spectral_d:.4f})")

# ========== 4. SATELLITE GEOMETRY ==========
print(f"\n{'='*70}")
print("4️⃣ SATELLITE GEOMETRY")
print("="*70)

sat_counts = []
high_snr_counts = []
for i in range(len(labels)):
    sat_receptions = dataset['satellite_receptions'][i]
    sat_counts.append(len(sat_receptions))
    high_snr = sum(1 for s in sat_receptions if s.get('ebno_db', 0) >= 12.0)
    high_snr_counts.append(high_snr)

print(f"  Mean satellites per sample: {np.mean(sat_counts):.1f}")
print(f"  Min satellites: {np.min(sat_counts)}")
print(f"  Mean high-SNR sats (≥12dB): {np.mean(high_snr_counts):.1f}")
print(f"  ✅ GOOD GEOMETRY" if np.mean(sat_counts) >= 10 else "  ⚠️  Limited satellites")

# ========== SUMMARY ==========
print(f"\n{'='*70}")
print("📋 SUMMARY")
print("="*70)

checks = []
checks.append(("Emitter altitude", max_alt < 1.0 if len(attack_altitudes) > 0 else False))
checks.append(("Power variance", attack_cv > 0.5))
checks.append(("Spectral separability", mean_spectral_d > 0.2))
checks.append(("Satellite geometry", np.mean(sat_counts) >= 10))

for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {check_name}")

all_passed = all(c[1] for c in checks)
print(f"\n{'='*70}")
if all_passed:
    print("🎉 ALL CHECKS PASSED! Dataset ready for training!")
    print("   Next steps:")
    print("   1. python3 scripts/train_stnn.py  # Train STNN localization")
    print("   2. python3 main.py                 # Train detector")
    print("   Expected performance:")
    print("   - STNN TDOA σ: 200-400 μs")
    print("   - STNN FDOA σ: 5-15 Hz")
    print("   - Detector AUC: 0.75-0.85")
else:
    print("⚠️  Some checks failed. Review dataset quality before training.")
print("="*70)
