#!/usr/bin/env python3
"""
Complete Pipeline Execution for Paper
======================================
Generates datasets (5000 samples each) and runs comprehensive tests.
"""
import os
import sys
import subprocess
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def wait_for_completion(process_name, timeout=3600):
    """Wait for a process to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = subprocess.run(['pgrep', '-f', process_name], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            return True  # Process completed
        time.sleep(10)
    return False  # Timeout

def generate_dataset(scenario, total_samples=5000):
    """Generate dataset for a scenario."""
    print_section(f"📡 Generating Scenario {scenario} Dataset ({total_samples} samples)")
    
    scenario_flag = 'sat' if scenario == 'A' else 'ground'
    
    cmd = [
        'python3', 'generate_dataset_parallel.py',
        '--scenario', scenario_flag,
        '--total-samples', str(total_samples),
        '--snr-list', '15,20',
        '--covert-amp-list', '0.5',
        '--samples-per-config', str(total_samples // 4),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Starting generation...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        # Find generated dataset
        dataset_dir = Path('dataset')
        if scenario == 'A':
            pattern = 'dataset_scenario_a_*.pkl'
        else:
            pattern = 'dataset_scenario_b_*.pkl'
        
        dataset_files = list(dataset_dir.glob(pattern))
        if dataset_files:
            latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
            print(f"✅ Dataset generated: {latest.name}")
            print(f"   Time: {elapsed/60:.1f} minutes")
            return latest
        else:
            print("⚠️  Generation completed but dataset not found")
            return None
    else:
        print(f"❌ Generation failed!")
        print(f"Error: {result.stderr[:500]}")
        return None

def analyze_dataset(dataset_path, scenario):
    """Analyze dataset and extract metrics."""
    print_section(f"📊 Analyzing Scenario {scenario} Dataset")
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        data = dataset.get('data', [])
        meta = dataset.get('meta', [])
        
        print(f"Dataset: {dataset_path.name}")
        print(f"Total samples: {len(data)}")
        print(f"Metadata entries: {len(meta)}")
        
        results = {
            'scenario': scenario,
            'dataset_file': dataset_path.name,
            'num_samples': len(data),
            'num_meta': len(meta),
        }
        
        if scenario == 'B':
            # Analyze EQ metrics
            preservations = []
            snr_improvements = []
            alpha_ratios = []
            injection_info_count = 0
            snr_raw_list = []
            snr_eq_list = []
            
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
                if 'eq_snr_raw_db' in meta_entry:
                    snr_raw_list.append(meta_entry['eq_snr_raw_db'])
                if 'eq_snr_eq_db' in meta_entry:
                    snr_eq_list.append(meta_entry['eq_snr_eq_db'])
            
            if preservations:
                results['preservation_median'] = float(np.median(preservations))
                results['preservation_mean'] = float(np.mean(preservations))
                results['preservation_std'] = float(np.std(preservations))
                results['preservation_q25'] = float(np.percentile(preservations, 25))
                results['preservation_q75'] = float(np.percentile(preservations, 75))
                results['preservation_ge_05'] = float(np.sum(np.array(preservations) >= 0.5) / len(preservations) * 100)
                
                print(f"\nEQ Performance Metrics:")
                print(f"  Pattern Preservation:")
                print(f"    Median: {results['preservation_median']:.3f}")
                print(f"    Mean: {results['preservation_mean']:.3f} ± {results['preservation_std']:.3f}")
                print(f"    Q25-Q75: {results['preservation_q25']:.3f} - {results['preservation_q75']:.3f}")
                print(f"    ≥0.5: {results['preservation_ge_05']:.1f}%")
            
            if snr_improvements:
                results['snr_improvement_mean'] = float(np.mean(snr_improvements))
                results['snr_improvement_std'] = float(np.std(snr_improvements))
                results['snr_improvement_median'] = float(np.median(snr_improvements))
                
                print(f"  SNR Improvement:")
                print(f"    Mean: {results['snr_improvement_mean']:.2f} ± {results['snr_improvement_std']:.2f} dB")
                print(f"    Median: {results['snr_improvement_median']:.2f} dB")
            
            if snr_raw_list and snr_eq_list:
                results['snr_raw_mean'] = float(np.mean(snr_raw_list))
                results['snr_eq_mean'] = float(np.mean(snr_eq_list))
                print(f"  SNR:")
                print(f"    Raw (before EQ): {results['snr_raw_mean']:.2f} dB")
                print(f"    EQ (after EQ): {results['snr_eq_mean']:.2f} dB")
            
            if alpha_ratios:
                results['alpha_ratio_mean'] = float(np.mean(alpha_ratios))
                results['alpha_ratio_in_range'] = float(np.sum((np.array(alpha_ratios) >= 0.1) & (np.array(alpha_ratios) <= 3.0)) / len(alpha_ratios) * 100)
                print(f"  Alpha Ratio:")
                print(f"    Mean: {results['alpha_ratio_mean']:.3f}x")
                print(f"    In range (0.1x-3x): {results['alpha_ratio_in_range']:.1f}%")
            
            print(f"  injection_info: {injection_info_count}/{len(meta)} samples ({injection_info_count/len(meta)*100:.1f}%)")
            results['injection_info_pct'] = injection_info_count / len(meta) * 100
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_tests():
    """Run pytest tests."""
    print_section("🧪 Running Tests")
    
    test_results = {}
    
    # Unit tests
    print("Running unit tests...")
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/unit/', '-v', '-m', 'unit', '--tb=short'],
        capture_output=True,
        text=True
    )
    test_results['unit'] = result.returncode == 0
    if result.returncode == 0:
        print("✅ Unit tests passed")
    else:
        print("⚠️  Some unit tests failed")
        print(result.stdout[-300:])
    
    # Integration tests
    print("\nRunning integration tests...")
    result = subprocess.run(
        ['python3', '-m', 'pytest', 'tests/integration/', '-v', '-m', 'integration', '--tb=short'],
        capture_output=True,
        text=True
    )
    test_results['integration'] = result.returncode == 0
    if result.returncode == 0:
        print("✅ Integration tests passed")
    else:
        print("⚠️  Some integration tests failed")
        print(result.stdout[-300:])
    
    return test_results

def generate_final_report(scenario_a_results, scenario_b_results, test_results):
    """Generate final comprehensive report."""
    print_section("📋 Final Report for Paper")
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scenario_a': scenario_a_results,
        'scenario_b': scenario_b_results,
        'tests': test_results,
    }
    
    print(f"Generated at: {report['timestamp']}\n")
    
    print("="*80)
    print("SCENARIO A (Single-hop Downlink)")
    print("="*80)
    if scenario_a_results:
        print(f"✅ Status: SUCCESS")
        print(f"📁 Dataset: {scenario_a_results.get('dataset_file', 'N/A')}")
        print(f"📊 Samples: {scenario_a_results.get('num_samples', 0)}")
    else:
        print("❌ Status: FAILED")
    
    print("\n" + "="*80)
    print("SCENARIO B (Dual-hop Relay)")
    print("="*80)
    if scenario_b_results:
        print(f"✅ Status: SUCCESS")
        print(f"📁 Dataset: {scenario_b_results.get('dataset_file', 'N/A')}")
        print(f"📊 Samples: {scenario_b_results.get('num_samples', 0)}")
        
        if 'preservation_median' in scenario_b_results:
            print(f"\n📈 EQ Performance:")
            print(f"   Pattern Preservation: {scenario_b_results['preservation_median']:.3f} (median)")
            print(f"   SNR Improvement: {scenario_b_results.get('snr_improvement_mean', 0):.2f} dB (mean)")
            print(f"   Alpha Ratio in Range: {scenario_b_results.get('alpha_ratio_in_range', 0):.1f}%")
            print(f"   Preservation ≥0.5: {scenario_b_results.get('preservation_ge_05', 0):.1f}%")
    else:
        print("❌ Status: FAILED")
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Unit Tests: {'✅ PASS' if test_results.get('unit') else '❌ FAIL'}")
    print(f"Integration Tests: {'✅ PASS' if test_results.get('integration') else '❌ FAIL'}")
    
    # Save report to file
    report_file = Path('result') / f"paper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(exist_ok=True)
    
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Also save as CSV for easy analysis
    if scenario_b_results:
        csv_data = []
        csv_data.append({
            'Metric': 'Pattern Preservation (median)',
            'Value': scenario_b_results.get('preservation_median', 0),
            'Unit': ''
        })
        csv_data.append({
            'Metric': 'SNR Improvement (mean)',
            'Value': scenario_b_results.get('snr_improvement_mean', 0),
            'Unit': 'dB'
        })
        csv_data.append({
            'Metric': 'Preservation ≥0.5',
            'Value': scenario_b_results.get('preservation_ge_05', 0),
            'Unit': '%'
        })
        csv_data.append({
            'Metric': 'Alpha Ratio in Range',
            'Value': scenario_b_results.get('alpha_ratio_in_range', 0),
            'Unit': '%'
        })
        
        csv_file = Path('result') / f"paper_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"📊 Metrics CSV saved to: {csv_file}")
    
    print("\n" + "="*80)
    print("✅ Pipeline completed!")
    print("="*80")

def main():
    """Main execution."""
    print_section("🚀 Complete Pipeline: Dataset Generation & Testing for Paper")
    
    print("Configuration:")
    print("  - Scenario A: 5000 samples")
    print("  - Scenario B: 5000 samples")
    print("  - SNR: 15, 20 dB")
    print("  - Covert amplitude: 0.5")
    print("\nEstimated time: 30-60 minutes\n")
    
    # Generate Scenario A
    scenario_a_dataset = generate_dataset('A', 5000)
    scenario_a_results = None
    if scenario_a_dataset:
        scenario_a_results = analyze_dataset(scenario_a_dataset, 'A')
    
    # Generate Scenario B
    scenario_b_dataset = generate_dataset('B', 5000)
    scenario_b_results = None
    if scenario_b_dataset:
        scenario_b_results = analyze_dataset(scenario_b_dataset, 'B')
    
    # Run tests
    test_results = run_tests()
    
    # Generate final report
    generate_final_report(scenario_a_results, scenario_b_results, test_results)

if __name__ == "__main__":
    main()

