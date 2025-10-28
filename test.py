import sys
sys.path.insert(0, '/mnt/project')
import numpy as np
import tensorflow as tf

# Simulate what happens in dataset generation
tx_pow = 1.0  # Example power
target_snr_db = 20.0
target_snr_linear = 10.0 ** (target_snr_db / 10.0)
sigma2_time = tx_pow / target_snr_linear

print(f"tx_pow: {tx_pow}")
print(f"target_snr_db: {target_snr_db}")
print(f"target_snr_linear: {target_snr_linear}")
print(f"sigma2_time: {sigma2_time}")
print(f"Expected SNR: {10*np.log10(tx_pow/sigma2_time):.2f} dB")

# Now check actual dataset
import pickle
dataset = pickle.load(open('dataset/dataset_samples1500_sats12.pkl', 'rb'))
rx_b = dataset['satellite_receptions'][1500][0]['rx_time_b_full']

signal_power = np.mean(np.abs(rx_b[:720])**2)
tail_power = np.mean(np.abs(rx_b[-200:])**2)

print(f"\nActual dataset:")
print(f"Signal power (first 720): {signal_power:.6e}")
print(f"Tail power (last 200): {tail_power:.6e}")
print(f"Measured SNR: {10*np.log10(signal_power/tail_power):.2f} dB")
print(f"Power ratio: {signal_power/tail_power:.2f}")