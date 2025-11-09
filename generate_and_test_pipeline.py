#!/usr/bin/env python3
"""
Complete Pipeline: Generate Datasets and Run Tests
==================================================
Generates datasets for both scenarios and runs comprehensive tests.
"""
import os
import sys
import subprocess
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_dataset_generation(scenario, expected_samples=5000):
    """Check if dataset was generated successfully."""
    dataset_dir = Path('dataset')
    if scenario == 'A':
        pattern = 'dataset_scenario_a_*.pkl'
    else:
        pattern = 'dataset_scenario_b_*.pkl'
    
    dataset_files = list(dataset_dir.glob(pattern))
    if len(dataset_files) == 0:
        return None, "No dataset files found"
    
    latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(latest, 'rb') as f:
            dataset = pickle.load(f)
        
        num_samples = len(dataset.get('data', []))
        return latest, num_samples
    except Exception as e:
        return None, f"Error loading dataset: {e}"

def generate_scenario_a():
    """Generate Scenario A dataset."""
    print_header("📡 Generating Scenario A Dataset (5000 samples)")
    
    cmd = [
        'python3', 'generate_dataset_parallel.py',
        '--scenario', 'sat',
        '--total-samples', '5000',
        '--snr-list', '15,20',
        '--covert-amp-list', '0.5',
        '--samples-per-config', '1250',
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting generation...\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        dataset_path, num_samples = check_dataset_generation('A', 5000)
        if dataset_path:
            print(f"✅ Scenario A dataset generated successfully!")
            print(f"   File: {dataset_path.name}")
            print(f"   Samples: {num_samples}")
            print(f"   Time: {elapsed/60:.1f} minutes")
            return True, dataset_path, num_samples
        else:
            print(f"⚠️  Generation completed but dataset not found")
            return False, None, 0
    else:
        print(f"❌ Scenario A generation failed!")
        print(f"Error: {result.stderr[:500]}")
        return False, None, 0

def generate_scenario_b():
    """Generate Scenario B dataset."""
    print_header("🌐 Generating Scenario B Dataset (5000 samples)")
    
    cmd = [
        'python3', 'generate_dataset_parallel.py',
        '--scenario', 'ground',
        '--total-samples', '5000',
        '--snr-list', '15,20',
        '--covert-amp-list', '0.5',
        '--samples-per-config', '1250',
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting generation...\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        dataset_path, num_samples = check_dataset_generation('B', 5000)
        if dataset_path:
            print(f"✅ Scenario B dataset generated successfully!")
            print(f"   File: {dataset_path.name}")
            print(f"   Samples: {num_samples}")
            print(f"   Time: {elapsed/60:.1f} minutes")
            return True, dataset_path, num_samples
        else:
            print(f"⚠️  Generation completed but dataset not found")
            return False, None, 0
    else:
        print(f"❌ Scenario B generation failed!")
        print(f"Error: {result.stderr[:500]}")
        return False, None, 0

def analyze_dataset(dataset_path, scenario):
    """Analyze dataset and extract metrics."""
    print_header(f"📊 Analyzing {scenario} Dataset")
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        data = dataset.get('data', [])
        meta = dataset.get('meta', [])
        
        print(f"Dataset: {dataset_path.name}")
        print(f"Total samples: {len(data)}")
        print(f"Metadata entries: {len(meta)}")
        
        if scenario == 'B':
            # Analyze EQ metrics
            preservations = []
            snr_improvements = []
            alpha_ratios = []
            injection_info_count = 0
            
            for meta_entry in meta:
                if isinstance(meta_entry, tuple):
                    _, meta_entry = meta_entry
                
                if 'eq_pattern_preservation' in meta_entry:
                    preservations.append(meta_entry['eq_pattern_preservation'])
                if 'eq_snr_improvement_db' in meta_entry:
                    snr_improvements.append(meta_entry['eq_snr_improvement_db'])
                if 'alpha_ratio' in meta_entry:
                    alpha_ratios.append(meta_entry['alpha_ratio'])
                if 'injection_info' in meta_entry:
                    injection_info_count += 1
            
            if preservations:
                print(f"\nEQ Performance Metrics:")
                print(f"  Pattern Preservation: median={np.median(preservations):.3f}, mean={np.mean(preservations):.3f}")
                print(f"  SNR Improvement: mean={np.mean(snr_improvements):.2f} dB, std={np.std(snr_improvements):.2f} dB")
                if alpha_ratios:
                    print(f"  Alpha Ratio: mean={np.mean(alpha_ratios):.3f}x")
                print(f"  injection_info: {injection_info_count}/{len(meta)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return False

def run_tests():
    """Run pytest tests."""
    print_header("🧪 Running Tests")
    
    # Run unit tests (fast)
    print("Running unit tests...")
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/unit/', '-v', '-m', 'unit'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Unit tests passed")
    else:
        print("⚠️  Some unit tests failed")
        print(result.stdout[-500:])
    
    # Run integration tests
    print("\nRunning integration tests...")
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/integration/', '-v', '-m', 'integration', '--tb=short'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Integration tests passed")
    else:
        print("⚠️  Some integration tests failed")
        print(result.stdout[-500:])
    
    return result.returncode == 0

def generate_report(scenario_a_result, scenario_b_result):
    """Generate final report."""
    print_header("📋 Final Report")
    
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Scenario A:")
    if scenario_a_result[0]:
        print(f"  ✅ Status: SUCCESS")
        print(f"  📁 Dataset: {scenario_a_result[1].name if scenario_a_result[1] else 'N/A'}")
        print(f"  📊 Samples: {scenario_a_result[2]}")
    else:
        print(f"  ❌ Status: FAILED")
    
    print("\nScenario B:")
    if scenario_b_result[0]:
        print(f"  ✅ Status: SUCCESS")
        print(f"  📁 Dataset: {scenario_b_result[1].name if scenario_b_result[1] else 'N/A'}")
        print(f"  📊 Samples: {scenario_b_result[2]}")
    else:
        print(f"  ❌ Status: FAILED")
    
    print("\n" + "="*80)

def main():
    """Main execution."""
    print_header("🚀 Complete Pipeline: Dataset Generation & Testing")
    
    # Generate Scenario A
    scenario_a_result = generate_scenario_a()
    if scenario_a_result[0] and scenario_a_result[1]:
        analyze_dataset(scenario_a_result[1], 'A')
    
    # Generate Scenario B
    scenario_b_result = generate_scenario_b()
    if scenario_b_result[0] and scenario_b_result[1]:
        analyze_dataset(scenario_b_result[1], 'B')
    
    # Run tests
    run_tests()
    
    # Generate final report
    generate_report(scenario_a_result, scenario_b_result)
    
    print("\n✅ Pipeline completed!")

if __name__ == "__main__":
    main()

