#!/usr/bin/env python3
"""
Generate Detailed Logs for 100 Samples
========================================
Logs detailed EQ metrics for diagnosis.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/workspace')

def generate_detailed_logs(num_samples=100):
    """Generate detailed logs for EQ diagnosis."""
    print("="*80)
    print("📊 Generating Detailed Logs for EQ Diagnosis")
    print("="*80)
    
    dataset_file = Path('dataset/dataset_scenario_b_5000.pkl')
    if not dataset_file.exists():
        print(f"❌ Dataset not found: {dataset_file}")
        return
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    meta = dataset.get('meta', [])
    
    # Select samples
    indices = np.random.choice(len(meta), min(num_samples, len(meta)), replace=False)
    
    logs = []
    
    for i, idx in enumerate(indices):
        meta_entry = meta[idx]
        if isinstance(meta_entry, tuple):
            _, meta_entry = meta_entry
        
        log_entry = {
            'sample_idx': idx,
            'snr_input_db': meta_entry.get('snr_db', 0),
            'snr_raw_db': meta_entry.get('eq_snr_raw_db', 0),
            'snr_eq_db': meta_entry.get('eq_snr_db', 0),
            'snr_improvement_db': meta_entry.get('eq_snr_improvement_db', 0),
            'alpha_used': meta_entry.get('eq_alpha', 0),
            'h_power_mean': meta_entry.get('csi_P_H', 0),
            'alpha_ratio': meta_entry.get('alpha_ratio', 0),
            'preservation_corr_eq': meta_entry.get('eq_pattern_preservation', 0),
            'pattern_indices_source': meta_entry.get('pattern_indices_source', 'unknown'),
            'target_subcarriers_count': meta_entry.get('target_subcarriers_count', 0),
        }
        
        # Get injection_info
        injection_info = meta_entry.get('injection_info', {})
        if injection_info:
            log_entry['subband_mode'] = injection_info.get('subband_mode', 'unknown')
            log_entry['pattern'] = injection_info.get('pattern', 'unknown')
            log_entry['num_covert_subs'] = injection_info.get('num_covert_subs', 0)
        
        logs.append(log_entry)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(indices)} samples...")
    
    # Create DataFrame
    df = pd.DataFrame(logs)
    
    # Save CSV
    output_file = Path('result') / f"eq_detailed_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 Summary Statistics")
    print("="*80)
    
    print(f"\nSNR Improvement:")
    print(f"   Mean: {df['snr_improvement_db'].mean():.2f} ± {df['snr_improvement_db'].std():.2f} dB")
    print(f"   Median: {df['snr_improvement_db'].median():.2f} dB")
    print(f"   Target: ≥ 5 dB")
    
    print(f"\nPattern Preservation:")
    print(f"   Mean: {df['preservation_corr_eq'].mean():.3f} ± {df['preservation_corr_eq'].std():.3f}")
    print(f"   Median: {df['preservation_corr_eq'].median():.3f}")
    print(f"   Target: ≥ 0.5")
    
    print(f"\nAlpha Ratio:")
    print(f"   Mean: {df['alpha_ratio'].mean():.3f}x")
    print(f"   In range (0.1x-3x): {(df['alpha_ratio'].between(0.1, 3.0).sum() / len(df) * 100):.1f}%")
    
    print(f"\nPattern Indices Source:")
    source_counts = df['pattern_indices_source'].value_counts()
    for source, count in source_counts.items():
        print(f"   {source}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n📄 Detailed logs saved to: {output_file}")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = generate_detailed_logs(100)

