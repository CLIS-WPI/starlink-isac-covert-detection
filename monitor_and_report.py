#!/usr/bin/env python3
"""
Monitor Scenario B generation and generate final report
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

def check_scenario_b_completion():
    """Check if Scenario B generation is complete."""
    # Check if process is running
    result = subprocess.run(['pgrep', '-f', 'generate_dataset_parallel.py.*ground'], 
                           capture_output=True, text=True)
    is_running = result.returncode == 0
    
    # Check for dataset file
    dataset_dir = Path('dataset')
    dataset_files = list(dataset_dir.glob('dataset_scenario_b_*.pkl'))
    
    if dataset_files:
        latest = max(dataset_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, 'rb') as f:
                dataset = pickle.load(f)
            num_samples = len(dataset.get('meta', []))
            return True, latest, num_samples, is_running
        except:
            return False, None, 0, is_running
    
    return False, None, 0, is_running

def analyze_scenario_b(dataset_path):
    """Analyze Scenario B dataset."""
    print("\n" + "="*80)
    print("📊 Analyzing Scenario B Dataset")
    print("="*80)
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        meta = dataset.get('meta', [])
        print(f"\nDataset: {dataset_path.name}")
        print(f"Total samples: {len(meta)}")
        
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
        
        results = {
            'num_samples': len(meta),
            'injection_info_pct': injection_info_count / len(meta) * 100 if len(meta) > 0 else 0,
        }
        
        if preservations:
            results['preservation_median'] = float(np.median(preservations))
            results['preservation_mean'] = float(np.mean(preservations))
            results['preservation_std'] = float(np.std(preservations))
            results['preservation_q25'] = float(np.percentile(preservations, 25))
            results['preservation_q75'] = float(np.percentile(preservations, 75))
            results['preservation_ge_05'] = float(np.sum(np.array(preservations) >= 0.5) / len(preservations) * 100)
            
            print(f"\n📈 Pattern Preservation:")
            print(f"   Median: {results['preservation_median']:.3f}")
            print(f"   Mean: {results['preservation_mean']:.3f} ± {results['preservation_std']:.3f}")
            print(f"   Q25-Q75: {results['preservation_q25']:.3f} - {results['preservation_q75']:.3f}")
            print(f"   ≥0.5: {results['preservation_ge_05']:.1f}%")
        
        if snr_improvements:
            results['snr_improvement_mean'] = float(np.mean(snr_improvements))
            results['snr_improvement_std'] = float(np.std(snr_improvements))
            results['snr_improvement_median'] = float(np.median(snr_improvements))
            
            print(f"\n📈 SNR Improvement:")
            print(f"   Mean: {results['snr_improvement_mean']:.2f} ± {results['snr_improvement_std']:.2f} dB")
            print(f"   Median: {results['snr_improvement_median']:.2f} dB")
        
        if snr_raw_list and snr_eq_list:
            results['snr_raw_mean'] = float(np.mean(snr_raw_list))
            results['snr_eq_mean'] = float(np.mean(snr_eq_list))
            
            print(f"\n📈 SNR:")
            print(f"   Raw (before EQ): {results['snr_raw_mean']:.2f} dB")
            print(f"   EQ (after EQ): {results['snr_eq_mean']:.2f} dB")
        
        if alpha_ratios:
            results['alpha_ratio_mean'] = float(np.mean(alpha_ratios))
            in_range = np.sum((np.array(alpha_ratios) >= 0.1) & (np.array(alpha_ratios) <= 3.0))
            results['alpha_ratio_in_range'] = float(in_range / len(alpha_ratios) * 100)
            
            print(f"\n📈 Alpha Ratio:")
            print(f"   Mean: {results['alpha_ratio_mean']:.3f}x")
            print(f"   In range (0.1x-3x): {results['alpha_ratio_in_range']:.1f}%")
        
        print(f"\n📈 Metadata:")
        print(f"   injection_info: {injection_info_count}/{len(meta)} ({results['injection_info_pct']:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_final_report(scenario_b_results):
    """Generate final report."""
    print("\n" + "="*80)
    print("📋 Final Report for Paper")
    print("="*80)
    
    print(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("="*80)
    print("SCENARIO A (Single-hop Downlink)")
    print("="*80)
    print("✅ Status: COMPLETED")
    print("📁 Dataset: dataset_scenario_a_5000.pkl")
    print("📊 Samples: 5000")
    
    print("\n" + "="*80)
    print("SCENARIO B (Dual-hop Relay)")
    print("="*80)
    if scenario_b_results:
        print("✅ Status: COMPLETED")
        print(f"📊 Samples: {scenario_b_results.get('num_samples', 0)}")
        
        if 'preservation_median' in scenario_b_results:
            print(f"\n📈 EQ Performance Metrics:")
            print(f"   Pattern Preservation (median): {scenario_b_results['preservation_median']:.3f}")
            print(f"   Pattern Preservation (mean): {scenario_b_results['preservation_mean']:.3f} ± {scenario_b_results['preservation_std']:.3f}")
            print(f"   Pattern Preservation (Q25-Q75): {scenario_b_results['preservation_q25']:.3f} - {scenario_b_results['preservation_q75']:.3f}")
            print(f"   Preservation ≥0.5: {scenario_b_results['preservation_ge_05']:.1f}%")
            
            if 'snr_improvement_mean' in scenario_b_results:
                print(f"   SNR Improvement (mean): {scenario_b_results['snr_improvement_mean']:.2f} ± {scenario_b_results['snr_improvement_std']:.2f} dB")
                print(f"   SNR Improvement (median): {scenario_b_results['snr_improvement_median']:.2f} dB")
            
            if 'alpha_ratio_in_range' in scenario_b_results:
                print(f"   Alpha Ratio in Range: {scenario_b_results['alpha_ratio_in_range']:.1f}%")
            
            if 'snr_raw_mean' in scenario_b_results:
                print(f"   SNR Raw: {scenario_b_results['snr_raw_mean']:.2f} dB")
                print(f"   SNR EQ: {scenario_b_results['snr_eq_mean']:.2f} dB")
    else:
        print("❌ Status: FAILED")
    
    # Save report
    report_file = Path('result') / f"paper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(exist_ok=True)
    
    import json
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scenario_a': {
            'status': 'completed',
            'dataset': 'dataset_scenario_a_5000.pkl',
            'samples': 5000,
        },
        'scenario_b': scenario_b_results or {},
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Save metrics CSV
    if scenario_b_results and 'preservation_median' in scenario_b_results:
        csv_data = []
        csv_data.append({
            'Metric': 'Pattern Preservation (median)',
            'Value': scenario_b_results['preservation_median'],
            'Unit': ''
        })
        csv_data.append({
            'Metric': 'Pattern Preservation (mean)',
            'Value': scenario_b_results['preservation_mean'],
            'Unit': ''
        })
        csv_data.append({
            'Metric': 'Pattern Preservation (std)',
            'Value': scenario_b_results['preservation_std'],
            'Unit': ''
        })
        csv_data.append({
            'Metric': 'Preservation ≥0.5',
            'Value': scenario_b_results['preservation_ge_05'],
            'Unit': '%'
        })
        if 'snr_improvement_mean' in scenario_b_results:
            csv_data.append({
                'Metric': 'SNR Improvement (mean)',
                'Value': scenario_b_results['snr_improvement_mean'],
                'Unit': 'dB'
            })
        if 'alpha_ratio_in_range' in scenario_b_results:
            csv_data.append({
                'Metric': 'Alpha Ratio in Range',
                'Value': scenario_b_results['alpha_ratio_in_range'],
                'Unit': '%'
            })
        
        csv_file = Path('result') / f"paper_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"📊 Metrics CSV saved to: {csv_file}")
    
    print("\n" + "="*80)
    print("✅ Report generation completed!")
    print("="*80)

def main():
    """Main monitoring loop."""
    print("="*80)
    print("🔍 Monitoring Scenario B Generation")
    print("="*80)
    
    max_wait = 3600  # 1 hour
    check_interval = 30  # 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        is_complete, dataset_path, num_samples, is_running = check_scenario_b_completion()
        
        if is_complete and num_samples >= 5000:
            print(f"\n✅ Scenario B generation completed!")
            print(f"   Dataset: {dataset_path.name}")
            print(f"   Samples: {num_samples}")
            
            # Analyze and generate report
            results = analyze_scenario_b(dataset_path)
            generate_final_report(results)
            return
        
        elif is_complete and num_samples < 5000:
            print(f"\n⚠️  Dataset found but incomplete: {num_samples}/5000 samples")
            print("   Waiting for completion...")
        
        elif is_running:
            elapsed = int(time.time() - start_time)
            print(f"\r⏳ Scenario B generation in progress... ({elapsed//60}m {elapsed%60}s)", end='', flush=True)
        
        else:
            print(f"\n⚠️  Generation process not found, checking for dataset...")
            if dataset_path:
                print(f"   Found dataset: {dataset_path.name} ({num_samples} samples)")
                if num_samples >= 5000:
                    results = analyze_scenario_b(dataset_path)
                    generate_final_report(results)
                    return
        
        time.sleep(check_interval)
    
    print(f"\n❌ Timeout waiting for Scenario B completion")
    print("   Checking for partial dataset...")
    
    is_complete, dataset_path, num_samples, _ = check_scenario_b_completion()
    if is_complete and dataset_path:
        print(f"   Found dataset: {dataset_path.name} ({num_samples} samples)")
        results = analyze_scenario_b(dataset_path)
        generate_final_report(results)

if __name__ == "__main__":
    main()

