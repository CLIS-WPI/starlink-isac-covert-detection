#!/usr/bin/env python3
"""
📋 Pattern Diversity Dataset Generation Wrapper
===============================================
Wrapper script to generate datasets with different injection patterns
without modifying core pipeline code.

Usage:
    python3 generate_pattern_diversity.py --pattern_type fixed --scenario B
    python3 generate_pattern_diversity.py --pattern_type random_sparse --scenario B
    python3 generate_pattern_diversity.py --pattern_type hopping --scenario B
    python3 generate_pattern_diversity.py --pattern_type phase_coded --scenario B
"""

import os
import sys
import argparse
import subprocess
import pickle
import numpy as np
from pathlib import Path

# Add wrapper for phase_coded pattern (modify injection function temporarily)
def inject_covert_channel_fixed_phase1_with_phase_coding(ofdm_np, resource_grid, 
                                                         pattern='fixed', 
                                                         subband_mode='mid',
                                                         selected_subcarriers=None,
                                                         selected_symbols=None,
                                                         covert_amp=0.5,
                                                         power_preserving=True,
                                                         phase_coded=False):
    """
    Wrapper that adds phase coding support to existing injection function.
    """
    from core.covert_injection_phase1 import inject_covert_channel_fixed_phase1
    
    # Call original function
    ofdm_np, injection_info = inject_covert_channel_fixed_phase1(
        ofdm_np, resource_grid,
        pattern=pattern,
        subband_mode=subband_mode,
        selected_subcarriers=selected_subcarriers,
        selected_symbols=selected_symbols,
        covert_amp=covert_amp,
        power_preserving=power_preserving
    )
    
    # Apply phase coding if requested
    if phase_coded:
        batch_size, num_rx, num_tx, num_symbols, num_subcarriers = ofdm_np.shape
        n_subs = resource_grid.num_effective_subcarriers
        
        # Get injection info to find selected subcarriers and symbols
        if injection_info and 'selected_subcarriers' in injection_info:
            selected_scs = injection_info['selected_subcarriers']
            selected_syms = injection_info.get('selected_symbols', np.arange(0, min(4, num_symbols)))
        else:
            # Default: middle band 24-39, symbols 0-3
            selected_scs = np.arange(24, min(40, n_subs))
            selected_syms = np.arange(0, min(4, num_symbols))
        
        # Apply random phase to each symbol
        for s in selected_syms:
            # Random phase for this symbol: [0, 2π)
            phi = np.random.uniform(0, 2 * np.pi)
            phase_factor = np.exp(1j * phi)
            
            # Apply to all selected subcarriers for this symbol
            for sc in selected_scs:
                if sc < num_subcarriers:
                    ofdm_np[0, 0, 0, s, sc] *= phase_factor
        
        # Update injection info
        if injection_info:
            injection_info['phase_coded'] = True
            injection_info['phase_values'] = [np.random.uniform(0, 2*np.pi) for _ in selected_syms]
    
    return ofdm_np, injection_info


def map_pattern_type_to_config(pattern_type):
    """
    Map pattern_type to (pattern, subband_mode) configuration.
    
    Args:
        pattern_type: 'fixed', 'random_sparse', 'hopping', 'phase_coded'
    
    Returns:
        tuple: (pattern, subband_mode, phase_coded_flag)
    """
    if pattern_type == 'fixed':
        return ('fixed', 'mid', False)
    elif pattern_type == 'random_sparse':
        return ('random', 'sparse', False)
    elif pattern_type == 'hopping':
        return ('fixed', 'hopping', False)
    elif pattern_type == 'phase_coded':
        return ('fixed', 'mid', True)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")


def apply_phase_coding_to_dataset(dataset_path, output_path=None):
    """
    Apply phase coding to dataset (post-processing for phase_coded pattern).
    
    Args:
        dataset_path: Path to input dataset
        output_path: Path to output dataset (if None, overwrite input)
    """
    print(f"\n🔄 Applying phase coding to dataset...")
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if 'tx_grids' not in dataset or 'labels' not in dataset:
        print("⚠️ Dataset missing tx_grids or labels, skipping phase coding")
        return False
    
    tx_grids = dataset['tx_grids']
    labels = dataset['labels']
    
    # Find attack samples
    attack_indices = np.where(labels == 1)[0]
    
    if len(attack_indices) == 0:
        print("⚠️ No attack samples found, skipping phase coding")
        return False
    
    # Apply phase coding to attack samples
    num_modified = 0
    for idx in attack_indices:
        tx_grid = np.array(tx_grids[idx])  # Make sure it's numpy array
        
        # Handle different shapes: (symbols, subcarriers) or (1, symbols, subcarriers) or (batch, symbols, subcarriers)
        while tx_grid.ndim > 2:
            tx_grid = tx_grid[0]  # Remove extra dimensions
        
        if tx_grid.ndim != 2:
            print(f"⚠️ Unexpected shape for tx_grid[{idx}]: {tx_grid.shape}, skipping")
            continue
        
        num_symbols, num_subcarriers = tx_grid.shape
        
        # Default injection pattern: subcarriers 24-39, symbols 0-3
        selected_scs = np.arange(24, min(40, num_subcarriers))
        selected_syms = np.arange(0, min(4, num_symbols))
        
        # Apply random phase to each symbol
        for s in selected_syms:
            # Random phase for this symbol: [0, 2π)
            phi = np.random.uniform(0, 2 * np.pi)
            phase_factor = np.exp(1j * phi)
            
            # Apply to all selected subcarriers for this symbol
            for sc in selected_scs:
                if sc < num_subcarriers:
                    tx_grid[s, sc] *= phase_factor
        
        # Update tx_grids
        if tx_grids[idx].ndim == 3:
            tx_grids[idx][0] = tx_grid
        else:
            tx_grids[idx] = tx_grid
        
        num_modified += 1
    
    print(f"  ✓ Applied phase coding to {num_modified} attack samples")
    
    # Save dataset
    if output_path is None:
        output_path = dataset_path
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"  ✓ Saved to: {output_path}")
    return True


def generate_dataset_with_pattern(pattern_type, scenario='B', total_samples=10000, output_name=None):
    """
    Generate dataset with specified pattern type.
    
    Args:
        pattern_type: 'fixed', 'random_sparse', 'hopping', 'phase_coded'
        scenario: 'A' or 'B'
        total_samples: Total number of samples
        output_name: Output filename (if None, auto-generate)
    """
    pattern, subband_mode, phase_coded = map_pattern_type_to_config(pattern_type)
    
    # Map scenario
    scenario_map = {'A': 'sat', 'B': 'ground'}
    scenario_mode = scenario_map.get(scenario.upper(), 'ground')
    
    # Generate output filename
    if output_name is None:
        output_name = f"dataset_scenario{scenario.upper()}_{pattern_type}.pkl"
    
    print("="*70)
    print(f"📋 Pattern Diversity Dataset Generation")
    print("="*70)
    print(f"Pattern Type: {pattern_type}")
    print(f"  → pattern: {pattern}")
    print(f"  → subband_mode: {subband_mode}")
    print(f"  → phase_coded: {phase_coded}")
    print(f"Scenario: {scenario_mode}")
    print(f"Total Samples: {total_samples}")
    print(f"Output: {output_name}")
    print("="*70)
    
    # Build command
    from config.settings import RESULT_DIR
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_output = os.path.join(RESULT_DIR, f"pattern_diversity_metadata_{pattern_type}.csv")
    
    cmd = [
        sys.executable,
        'generate_dataset_parallel.py',
        '--scenario', scenario_mode,
        '--total-samples', str(total_samples),
        '--pattern', pattern,
        '--subband', subband_mode,
        '--output-csv', csv_output  # Fix CSV output path issue
    ]
    
    # Note: generate_dataset_parallel.py doesn't support --output flag
    # It auto-generates filename based on scenario and samples
    # We'll rename the file after generation
    
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    
    # Run dataset generation
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Dataset generation failed with exit code {result.returncode}")
        return False
    
    # Find generated dataset file (auto-named by generate_dataset_parallel.py)
    from config.settings import DATASET_DIR
    import glob
    
    # Map scenario_mode to file pattern
    # 'sat' -> 'a', 'ground' -> 'b'
    scenario_file_map = {'sat': 'a', 'ground': 'b'}
    scenario_file_char = scenario_file_map.get(scenario_mode, 'b')
    
    # Pattern: dataset_scenario_b_*.pkl or dataset_scenario_a_*.pkl
    pattern_files = glob.glob(os.path.join(DATASET_DIR, f"dataset_scenario_{scenario_file_char}_*.pkl"))
    pattern_files.sort(key=os.path.getmtime, reverse=True)  # Most recent first
    
    if not pattern_files:
        print(f"❌ Generated dataset file not found in {DATASET_DIR}")
        return False
    
    generated_file = pattern_files[0]
    output_path = os.path.join(DATASET_DIR, output_name)
    
    # Post-process for phase_coded
    if phase_coded:
        if os.path.exists(generated_file):
            apply_phase_coding_to_dataset(generated_file, output_path)
            # Remove original file
            if generated_file != output_path:
                os.remove(generated_file)
        else:
            print(f"⚠️ Generated file not found: {generated_file}")
            return False
    else:
        # Rename to desired output name
        if generated_file != output_path:
            import shutil
            shutil.move(generated_file, output_path)
    
    print(f"\n✅ Dataset generated: {output_name}")
    return True


def calculate_power_diff(dataset_path):
    """Calculate power difference from dataset."""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if 'tx_grids' not in dataset or 'labels' not in dataset:
        return 0.0
    
    tx_grids = dataset['tx_grids']
    labels = dataset['labels']
    
    attack_indices = np.where(labels == 1)[0]
    benign_indices = np.where(labels == 0)[0]
    
    if len(attack_indices) > 0 and len(benign_indices) > 0:
        # Sample subset for efficiency
        n_samples = min(100, len(attack_indices), len(benign_indices))
        attack_tx_power = np.mean([np.mean(np.abs(tx_grids[i])**2) for i in attack_indices[:n_samples]])
        benign_tx_power = np.mean([np.mean(np.abs(tx_grids[i])**2) for i in benign_indices[:n_samples]])
        power_diff = abs(attack_tx_power - benign_tx_power) / (benign_tx_power + 1e-12) * 100.0
        return power_diff
    return 0.0


def evaluate_pattern_dataset(dataset_path, scenario='B'):
    """
    Evaluate dataset using CNN detector.
    
    Args:
        dataset_path: Path to dataset .pkl file
        scenario: 'A' or 'B'
    
    Returns:
        dict: Metrics (auc_mean, auc_std, precision, recall, f1, power_diff)
    """
    print(f"\n📊 Evaluating dataset: {os.path.basename(dataset_path)}")
    
    # Calculate power difference
    power_diff = calculate_power_diff(dataset_path)
    
    # Map scenario
    scenario_map = {'A': 'sat', 'B': 'ground'}
    scenario_mode = scenario_map.get(scenario.upper(), 'ground')
    
    # Copy dataset to expected location for evaluation
    from config.settings import DATASET_DIR, RESULT_DIR
    import shutil
    import glob
    
    # Find expected dataset name pattern
    dataset_basename = os.path.basename(dataset_path)
    temp_dataset_path = os.path.join(DATASET_DIR, dataset_basename)
    
    # Copy if different location
    if dataset_path != temp_dataset_path:
        shutil.copy2(dataset_path, temp_dataset_path)
    
    try:
        # Run CNN evaluation
        cmd = [
            sys.executable,
            'main_detection_cnn.py',
            '--scenario', scenario_mode,
            '--epochs', '30',
            '--batch-size', '512'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        # Parse results from JSON
        result_file = os.path.join(RESULT_DIR, f"scenario_{scenario_mode[0]}", "detection_results_cnn.json")
        
        if os.path.exists(result_file):
            import json
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            metrics = {
                'auc_mean': results.get('auc', 0.0),
                'auc_std': 0.0,  # Single-split, no std
                'precision': results.get('precision', 0.0),
                'recall': results.get('recall', 0.0),
                'f1': results.get('f1', 0.0),
                'power_diff': power_diff
            }
        else:
            print(f"⚠️ Result file not found: {result_file}")
            metrics = {
                'auc_mean': 0.0,
                'auc_std': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'power_diff': power_diff
            }
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        metrics = {
            'auc_mean': 0.0,
            'auc_std': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'power_diff': power_diff
        }
    finally:
        # Cleanup: restore original dataset if needed
        # (Keep temp copy for now, will be overwritten by next evaluation)
        pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Pattern Diversity Dataset Generation')
    parser.add_argument('--pattern_type', type=str, required=False,
                       choices=['fixed', 'random_sparse', 'hopping', 'phase_coded'],
                       help='Pattern type to generate (required if --all not used)')
    parser.add_argument('--scenario', type=str, default='B',
                       choices=['A', 'B'],
                       help='Scenario A (sat) or B (ground)')
    parser.add_argument('--total-samples', type=int, default=10000,
                       help='Total number of samples')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (auto-generated if not provided)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate dataset after generation')
    parser.add_argument('--all', action='store_true',
                       help='Generate all pattern types')
    
    args = parser.parse_args()
    
    # Validate: either --all or --pattern_type must be provided
    if not args.all and not args.pattern_type:
        parser.error("Either --all or --pattern_type must be provided")
    
    if args.all:
        # Generate all patterns
        patterns = ['fixed', 'random_sparse', 'hopping', 'phase_coded']
        results = []
        
        for pattern_type in patterns:
            print(f"\n{'='*70}")
            print(f"Generating {pattern_type}...")
            print(f"{'='*70}\n")
            
            success = generate_dataset_with_pattern(
                pattern_type=pattern_type,
                scenario=args.scenario,
                total_samples=args.total_samples,
                output_name=None
            )
            
            if success and args.evaluate:
                dataset_path = f"dataset/dataset_scenario{args.scenario.upper()}_{pattern_type}.pkl"
                if os.path.exists(dataset_path):
                    metrics = evaluate_pattern_dataset(dataset_path, args.scenario)
                    results.append({
                        'pattern_type': pattern_type,
                        **metrics
                    })
        
        # Save results to CSV
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            # Format: pattern_type, auc_mean, auc_std, power_diff
            csv_df = df[['pattern_type', 'auc_mean', 'auc_std', 'power_diff']].copy()
            csv_path = f"result/pattern_diversity_results_scenario{args.scenario.upper()}.csv"
            os.makedirs('result', exist_ok=True)
            csv_df.to_csv(csv_path, index=False, float_format='%.3f')
            print(f"\n✅ Results saved to: {csv_path}")
            print("\n📊 Pattern Diversity Results:")
            print(csv_df.to_string(index=False))
    else:
        # Generate single pattern
        success = generate_dataset_with_pattern(
            pattern_type=args.pattern_type,
            scenario=args.scenario,
            total_samples=args.total_samples,
            output_name=args.output
        )
        
        if success and args.evaluate:
            if args.output:
                dataset_path = args.output
            else:
                dataset_path = f"dataset/dataset_scenario{args.scenario.upper()}_{args.pattern_type}.pkl"
            
            if os.path.exists(dataset_path):
                metrics = evaluate_pattern_dataset(dataset_path, args.scenario)
                print(f"\n📊 Results:")
                print(f"  AUC: {metrics['auc']:.3f}")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  F1: {metrics['f1']:.3f}")
                print(f"  Power Diff: {metrics['power_diff']:.4f}%")


if __name__ == "__main__":
    main()

