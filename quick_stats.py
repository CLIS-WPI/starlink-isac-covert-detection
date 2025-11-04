#!/usr/bin/env python3
"""
Quick Dataset Statistics Summary
=================================
Shows a quick summary of dataset statistics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np

def quick_summary(dataset_path):
    """Print quick dataset summary."""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    labels = dataset['labels']
    n_benign = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    
    print(f"\n📊 Quick Summary: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")
    print(f"  Total:  {len(labels)} samples")
    print(f"  Benign: {n_benign} ({n_benign/len(labels)*100:.1f}%)")
    print(f"  Attack: {n_attack} ({n_attack/len(labels)*100:.1f}%)")
    
    if 'tx_grids' in dataset:
        tx_grids = dataset['tx_grids']
        benign_powers = [np.mean(np.abs(np.squeeze(tx_grids[i]))**2) 
                         for i in range(len(labels)) if labels[i] == 0]
        attack_powers = [np.mean(np.abs(np.squeeze(tx_grids[i]))**2) 
                         for i in range(len(labels)) if labels[i] == 1]
        
        benign_mean = np.mean(benign_powers)
        attack_mean = np.mean(attack_powers)
        rel_diff = abs(attack_mean - benign_mean) / (benign_mean + 1e-12) * 100
        
        print(f"\n  Power Difference: {rel_diff:.2f}%", end=" ")
        if 5.0 <= rel_diff <= 15.0:
            print("✅")
        elif rel_diff < 5.0:
            print("⚠️ (too low)")
        else:
            print("⚠️ (too high)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset_samples200_sats12.pkl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Not found: {dataset_path}")
        sys.exit(1)
    
    quick_summary(dataset_path)
