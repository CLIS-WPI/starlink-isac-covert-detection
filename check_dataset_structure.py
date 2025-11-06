#!/usr/bin/env python3
"""Quick check of dataset structure"""
import pickle
from pathlib import Path

dataset_path = Path("dataset/dataset_samples500_sats12.pkl")
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

print("Dataset keys:", list(data.keys()))
print("\nDataset structure:")
for key, value in data.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, dict):
        print(f"  {key}: dict with keys {list(value.keys())}")
    else:
        print(f"  {key}: type={type(value)}")
