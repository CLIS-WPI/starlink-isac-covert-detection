#!/usr/bin/env python3
"""
🔬 Ablation Study Framework
============================
Systematic testing of different configurations to find optimal settings.

Test configurations:
1. Baseline (current semi-fixed)
2. +CSI Fusion
3. +ResNet architecture
4. +Spectrogram features
5. +Focal loss
6. +All combined

Each test runs 3 times for statistical significance.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import pickle
import random
import tensorflow as tf

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Test configurations
ABLATION_CONFIGS = {
    "baseline": {
        "name": "Baseline (Semi-Fixed)",
        "CSI_FUSION": False,
        "USE_SPECTROGRAM": False,
        "USE_RESIDUAL_CNN": False,
        "USE_FOCAL_LOSS": False,
        "NUM_SAMPLES_PER_CLASS": 1500,
    },
    "csi": {
        "name": "+CSI Fusion",
        "CSI_FUSION": True,
        "USE_SPECTROGRAM": False,
        "USE_RESIDUAL_CNN": False,
        "USE_FOCAL_LOSS": False,
        "NUM_SAMPLES_PER_CLASS": 1500,
    },
    "resnet": {
        "name": "+ResNet Architecture",
        "CSI_FUSION": False,
        "USE_SPECTROGRAM": False,
        "USE_RESIDUAL_CNN": True,
        "USE_FOCAL_LOSS": False,
        "NUM_SAMPLES_PER_CLASS": 1500,
    },
    "spectrogram": {
        "name": "+Spectrogram (STFT)",
        "CSI_FUSION": False,
        "USE_SPECTROGRAM": True,
        "USE_RESIDUAL_CNN": False,
        "USE_FOCAL_LOSS": False,
        "NUM_SAMPLES_PER_CLASS": 1500,
    },
    "focal_loss": {
        "name": "+Focal Loss",
        "CSI_FUSION": False,
        "USE_SPECTROGRAM": False,
        "USE_RESIDUAL_CNN": False,
        "USE_FOCAL_LOSS": True,
        "NUM_SAMPLES_PER_CLASS": 1500,
    },
    "more_data": {
        "name": "+More Data (3000)",
        "CSI_FUSION": False,
        "USE_SPECTROGRAM": False,
        "USE_RESIDUAL_CNN": False,
        "USE_FOCAL_LOSS": False,
        "NUM_SAMPLES_PER_CLASS": 3000,
    },
    "full": {
        "name": "Full (All Features)",
        "CSI_FUSION": True,
        "USE_SPECTROGRAM": True,
        "USE_RESIDUAL_CNN": True,
        "USE_FOCAL_LOSS": True,
        "NUM_SAMPLES_PER_CLASS": 3000,
    },
}


def update_config(config_dict):
    """Update config/settings.py with test configuration"""
    
    settings_path = "config/settings.py"
    
    # Read current settings
    with open(settings_path, 'r') as f:
        lines = f.readlines()
    
    # Update specific lines
    updated_lines = []
    for line in lines:
        if line.strip().startswith("CSI_FUSION ="):
            updated_lines.append(f"CSI_FUSION = {config_dict['CSI_FUSION']}\n")
        elif line.strip().startswith("USE_SPECTROGRAM ="):
            updated_lines.append(f"USE_SPECTROGRAM = {config_dict['USE_SPECTROGRAM']}\n")
        elif line.strip().startswith("USE_RESIDUAL_CNN ="):
            updated_lines.append(f"USE_RESIDUAL_CNN = {config_dict['USE_RESIDUAL_CNN']}\n")
        elif line.strip().startswith("USE_FOCAL_LOSS ="):
            updated_lines.append(f"USE_FOCAL_LOSS = {config_dict['USE_FOCAL_LOSS']}\n")
        elif line.strip().startswith("NUM_SAMPLES_PER_CLASS ="):
            updated_lines.append(f"NUM_SAMPLES_PER_CLASS = {config_dict['NUM_SAMPLES_PER_CLASS']}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(settings_path, 'w') as f:
        f.writelines(updated_lines)


def run_experiment(config_name, config_dict, run_id):
    """Run single experiment with given configuration"""
    
    print("\n" + "="*80)
    print(f"🔬 EXPERIMENT: {config_dict['name']} (Run {run_id}/3)")
    print("="*80)
    
    # Update configuration
    print(f"\n⚙️  Updating configuration...")
    update_config(config_dict)
    
    # Verify config
    print(f"✓ Configuration updated")
    for key, value in config_dict.items():
        if key != "name":
            print(f"  {key}: {value}")
    
    # Generate dataset (if needed)
    dataset_path = f"dataset/dataset_samples{config_dict['NUM_SAMPLES_PER_CLASS']}_sats12.pkl"
    
    if not Path(dataset_path).exists():
        print(f"\n📊 Generating dataset...")
        os.system("python3 generate_dataset_parallel.py")
    else:
        print(f"\n✓ Dataset exists: {dataset_path}")
    
    # Train CNN
    print(f"\n🧠 Training CNN...")
    start_time = time.time()
    
    result = os.system("python3 main_detection_cnn.py --epochs 30 2>&1 | tee temp_log.txt")
    
    training_time = time.time() - start_time
    
    if result != 0:
        print(f"❌ Training failed!")
        return None
    
    # Parse results
    try:
        with open("result/detection_results_cnn.json", 'r') as f:
            results = json.load(f)
        
        metrics = results.get('metrics', {})
        
        experiment_results = {
            'config_name': config_name,
            'config': config_dict,
            'run_id': run_id,
            'auc': metrics.get('auc', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'training_time': training_time,
            'power_diff': results.get('power_analysis', {}).get('difference_pct', 0),
        }
        
        print(f"\n📊 Results:")
        print(f"  AUC:       {experiment_results['auc']:.4f}")
        print(f"  Precision: {experiment_results['precision']:.4f}")
        print(f"  Recall:    {experiment_results['recall']:.4f}")
        print(f"  F1:        {experiment_results['f1']:.4f}")
        print(f"  Time:      {training_time/60:.1f} min")
        
        return experiment_results
        
    except Exception as e:
        print(f"❌ Failed to parse results: {e}")
        return None


def run_ablation_study(configs_to_test=None, num_runs=3):
    """Run full ablation study"""
    
    if configs_to_test is None:
        configs_to_test = list(ABLATION_CONFIGS.keys())
    
    print("\n" + "="*80)
    print("🔬 ABLATION STUDY - Systematic Configuration Testing")
    print("="*80)
    print(f"\nConfigurations to test: {len(configs_to_test)}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total experiments: {len(configs_to_test) * num_runs}")
    
    for config_name in configs_to_test:
        print(f"\n  - {ABLATION_CONFIGS[config_name]['name']}")
    
    input("\nPress Enter to start ablation study...")
    
    # Run experiments
    all_results = []
    
    for config_name in configs_to_test:
        config_dict = ABLATION_CONFIGS[config_name]
        
        config_results = []
        
        for run_id in range(1, num_runs + 1):
            result = run_experiment(config_name, config_dict, run_id)
            
            if result:
                config_results.append(result)
                all_results.append(result)
        
        # Aggregate results for this config
        if config_results:
            avg_auc = np.mean([r['auc'] for r in config_results])
            std_auc = np.std([r['auc'] for r in config_results])
            
            print(f"\n📊 Aggregate Results for {config_dict['name']}:")
            print(f"  AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Save all results
    results_file = f"result/ablation_study_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ All results saved to {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 ABLATION STUDY SUMMARY")
    print("="*80)
    
    for config_name in configs_to_test:
        config_results = [r for r in all_results if r['config_name'] == config_name]
        
        if config_results:
            aucs = [r['auc'] for r in config_results]
            avg_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            config_dict = ABLATION_CONFIGS[config_name]
            print(f"\n{config_dict['name']}:")
            print(f"  AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Find best
    config_means = {}
    for config_name in configs_to_test:
        config_results = [r for r in all_results if r['config_name'] == config_name]
        if config_results:
            config_means[config_name] = np.mean([r['auc'] for r in config_results])
    
    if config_means:
        best_config = max(config_means, key=config_means.get)
        print(f"\n🏆 Best Configuration: {ABLATION_CONFIGS[best_config]['name']}")
        print(f"   AUC: {config_means[best_config]:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--configs', nargs='+', 
                       help='Specific configs to test (default: all)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per config (default: 3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (only baseline, csi, full with 1 run)')
    
    args = parser.parse_args()
    
    if args.quick:
        configs = ['baseline', 'csi', 'full']
        runs = 1
        print("🚀 Quick ablation study mode")
    else:
        configs = args.configs if args.configs else None
        runs = args.runs
    
    run_ablation_study(configs, runs)
