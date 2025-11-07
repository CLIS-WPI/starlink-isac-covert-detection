#!/usr/bin/env python3
"""
Regenerate metadata CSV from dataset .pkl file
==============================================
Extracts metadata from dataset and exports to CSV with all Phase 6 fields.
"""

import os
import sys
import pickle
import pandas as pd
import argparse

from config.settings import DATASET_DIR, RESULT_DIR


def regenerate_csv(dataset_path, output_csv=None):
    """
    Regenerate metadata CSV from dataset.
    
    Args:
        dataset_path: Path to dataset .pkl file
        output_csv: Output CSV path (default: auto-detect from dataset name)
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Auto-detect output CSV path if not provided
    if output_csv is None:
        dataset_name = os.path.basename(dataset_path)
        if 'scenario_a' in dataset_name:
            output_csv = f"{RESULT_DIR}/dataset_metadata_phase1_scenario_a.csv"
        elif 'scenario_b' in dataset_name:
            output_csv = f"{RESULT_DIR}/dataset_metadata_phase1_scenario_b.csv"
        else:
            output_csv = f"{RESULT_DIR}/dataset_metadata_phase1.csv"
    
    print("="*70)
    print("🔄 Regenerating Metadata CSV")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_csv}")
    print("="*70)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract metadata
    print("📊 Extracting metadata...")
    meta = dataset.get('meta', [])
    labels = dataset.get('labels', [])
    
    if not meta:
        print("❌ No metadata found in dataset")
        return False
    
    # Convert to DataFrame
    metadata_rows = []
    for i, meta_dict in enumerate(meta):
        if isinstance(meta_dict, dict):
            row = {
                'sample_idx': i,
                'label': int(labels[i]) if i < len(labels) else None,
                **meta_dict  # Include all meta fields (including Phase 6)
            }
            metadata_rows.append(row)
    
    if not metadata_rows:
        print("❌ No valid metadata rows found")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(metadata_rows)
    
    # Check Phase 6 columns
    phase6_cols = ['fd_ul', 'fd_dl', 'G_r_mean', 'delay_samples', 'snr_ul', 'snr_dl']
    found_cols = [col for col in phase6_cols if col in df.columns]
    missing_cols = [col for col in phase6_cols if col not in df.columns]
    
    print(f"\n✅ Metadata extracted: {len(df)} rows, {len(df.columns)} columns")
    if found_cols:
        print(f"✅ Phase 6 columns found: {found_cols}")
    if missing_cols:
        print(f"⚠️  Phase 6 columns missing: {missing_cols}")
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV saved to: {output_csv}")
    print(f"   Columns: {list(df.columns)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate metadata CSV from dataset")
    parser.add_argument('--dataset', type=str, default=None,
                       help="Path to dataset .pkl file (default: auto-detect scenario)")
    parser.add_argument('--scenario', type=str, choices=['a', 'b', 'sat', 'ground'], default=None,
                       help="Scenario to regenerate (a/sat or b/ground)")
    parser.add_argument('--output', type=str, default=None,
                       help="Output CSV path (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    elif args.scenario:
        scenario_name = 'scenario_a' if args.scenario in ['a', 'sat'] else 'scenario_b'
        dataset_path = f"{DATASET_DIR}/dataset_{scenario_name}.pkl"
    else:
        # Try to auto-detect
        if os.path.exists(f"{DATASET_DIR}/dataset_scenario_b.pkl"):
            dataset_path = f"{DATASET_DIR}/dataset_scenario_b.pkl"
            print("ℹ️  Auto-detected: Scenario B")
        elif os.path.exists(f"{DATASET_DIR}/dataset_scenario_a.pkl"):
            dataset_path = f"{DATASET_DIR}/dataset_scenario_a.pkl"
            print("ℹ️  Auto-detected: Scenario A")
        else:
            print("❌ No dataset found. Please specify --dataset or --scenario")
            return 1
    
    # Regenerate CSV
    success = regenerate_csv(dataset_path, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

