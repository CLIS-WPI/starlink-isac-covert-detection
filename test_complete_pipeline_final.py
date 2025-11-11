#!/usr/bin/env python3
"""
Complete Pipeline Test - Final Verification
===========================================
Tests complete pipeline for both Scenario A and B from scratch.
"""
import os
import sys
import subprocess
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, timeout=600):
    """Run a command and return success status."""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr[:500]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False, str(e)

def verify_dataset(scenario_name, expected_samples=5000):
    """Verify dataset exists and is valid."""
    dataset_file = Path(f'dataset/dataset_{scenario_name}_5000.pkl')
    
    if not dataset_file.exists():
        return False, f"Dataset file not found: {dataset_file}"
    
    try:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        
        if len(dataset.get('labels', [])) != expected_samples:
            return False, f"Expected {expected_samples} samples, got {len(dataset.get('labels', []))}"
        
        if 'tx_grids' not in dataset or 'rx_grids' not in dataset:
            return False, "Missing required fields (tx_grids or rx_grids)"
        
        size_mb = dataset_file.stat().st_size / (1024**2)
        return True, f"Dataset valid: {len(dataset['labels'])} samples, {size_mb:.2f} MB"
    except Exception as e:
        return False, f"Error loading dataset: {e}"

def verify_training_results(scenario_name):
    """Verify training results exist."""
    result_file = Path(f'result/{scenario_name}/detection_results_cnn.json')
    
    if not result_file.exists():
        return False, f"Results file not found: {result_file}"
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        if 'metrics' not in results:
            return False, "Missing metrics in results"
        
        metrics = results['metrics']
        auc = metrics.get('auc', 0)
        
        return True, f"Results valid: AUC = {auc:.4f}"
    except Exception as e:
        return False, f"Error loading results: {e}"

def analyze_results(scenario_name):
    """Analyze results for a scenario."""
    dataset_file = Path(f'dataset/dataset_{scenario_name}_5000.pkl')
    result_file = Path(f'result/{scenario_name}/detection_results_cnn.json')
    
    if not dataset_file.exists() or not result_file.exists():
        return None
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    metrics = results.get('metrics', {})
    meta = dataset.get('meta', [])
    
    # Analyze EQ for Scenario B
    preservations = []
    snr_improvements = []
    for m in meta[:1000]:
        if isinstance(m, tuple):
            _, m = m
        if 'eq_pattern_preservation' in m:
            preservations.append(m['eq_pattern_preservation'])
        if 'eq_snr_improvement_db' in m:
            snr_improvements.append(m['eq_snr_improvement_db'])
    
    analysis = {
        'dataset_size': len(dataset['labels']),
        'dataset_size_mb': dataset_file.stat().st_size / (1024**2),
        'auc': metrics.get('auc', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
    }
    
    if preservations:
        analysis['pattern_preservation'] = np.median(preservations)
    if snr_improvements:
        analysis['snr_improvement'] = np.mean(snr_improvements)
    
    return analysis

def main():
    """Run complete pipeline test."""
    print("="*80)
    print("🧪 COMPLETE PIPELINE TEST - FINAL VERIFICATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'scenario_a': {},
        'scenario_b': {},
        'status': 'in_progress'
    }
    
    # ===== Scenario A =====
    print("\n" + "="*80)
    print("1️⃣  SCENARIO A (Single-hop Downlink)")
    print("="*80)
    
    # Generate dataset
    success, output = run_command(
        "python3 generate_dataset_parallel.py --scenario sat --total-samples 5000",
        "Scenario A: Dataset Generation",
        timeout=1800
    )
    results['scenario_a']['dataset_generation'] = success
    
    if success:
        # Copy to standard name
        subprocess.run("cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl 2>/dev/null || true", shell=True)
        
        # Verify dataset
        valid, msg = verify_dataset('scenario_a', 5000)
        print(f"  Dataset verification: {'✅' if valid else '❌'} {msg}")
        results['scenario_a']['dataset_valid'] = valid
        
        if valid:
            # Train model
            success, output = run_command(
                "python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512",
                "Scenario A: Model Training",
                timeout=1800
            )
            results['scenario_a']['training'] = success
            
            if success:
                # Verify results
                valid, msg = verify_training_results('scenario_a')
                print(f"  Results verification: {'✅' if valid else '❌'} {msg}")
                results['scenario_a']['results_valid'] = valid
                
                # Analyze
                analysis = analyze_results('scenario_a')
                if analysis:
                    results['scenario_a']['analysis'] = analysis
                    print(f"\n  📊 Analysis:")
                    print(f"     AUC: {analysis['auc']:.4f}")
                    print(f"     Precision: {analysis['precision']:.4f}")
                    print(f"     Recall: {analysis['recall']:.4f}")
                    print(f"     F1: {analysis['f1']:.4f}")
    
    # ===== Scenario B =====
    print("\n" + "="*80)
    print("2️⃣  SCENARIO B (Dual-hop Relay)")
    print("="*80)
    
    # Generate dataset
    success, output = run_command(
        "python3 generate_dataset_parallel.py --scenario ground --total-samples 5000",
        "Scenario B: Dataset Generation",
        timeout=1800
    )
    results['scenario_b']['dataset_generation'] = success
    
    if success:
        # Copy to standard name
        subprocess.run("cp dataset/dataset_scenario_b_4998.pkl dataset/dataset_scenario_b_5000.pkl 2>/dev/null || true", shell=True)
        
        # Verify dataset
        valid, msg = verify_dataset('scenario_b', 5000)
        print(f"  Dataset verification: {'✅' if valid else '❌'} {msg}")
        results['scenario_b']['dataset_valid'] = valid
        
        if valid:
            # Train model
            success, output = run_command(
                "python3 main_detection_cnn.py --scenario ground --epochs 30 --batch-size 512",
                "Scenario B: Model Training",
                timeout=1800
            )
            results['scenario_b']['training'] = success
            
            if success:
                # Verify results
                valid, msg = verify_training_results('scenario_b')
                print(f"  Results verification: {'✅' if valid else '❌'} {msg}")
                results['scenario_b']['results_valid'] = valid
                
                # Analyze
                analysis = analyze_results('scenario_b')
                if analysis:
                    results['scenario_b']['analysis'] = analysis
                    print(f"\n  📊 Analysis:")
                    print(f"     AUC: {analysis['auc']:.4f}")
                    print(f"     Precision: {analysis['precision']:.4f}")
                    print(f"     Recall: {analysis['recall']:.4f}")
                    print(f"     F1: {analysis['f1']:.4f}")
                    if 'pattern_preservation' in analysis:
                        print(f"     Pattern Preservation: {analysis['pattern_preservation']:.3f}")
                    if 'snr_improvement' in analysis:
                        print(f"     SNR Improvement: {analysis['snr_improvement']:.2f} dB")
    
    # ===== Final Report =====
    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    
    # Check overall status
    scenario_a_ok = (
        results['scenario_a'].get('dataset_generation', False) and
        results['scenario_a'].get('dataset_valid', False) and
        results['scenario_a'].get('training', False) and
        results['scenario_a'].get('results_valid', False)
    )
    
    scenario_b_ok = (
        results['scenario_b'].get('dataset_generation', False) and
        results['scenario_b'].get('dataset_valid', False) and
        results['scenario_b'].get('training', False) and
        results['scenario_b'].get('results_valid', False)
    )
    
    print(f"\n✅ Scenario A: {'PASS' if scenario_a_ok else 'FAIL'}")
    print(f"✅ Scenario B: {'PASS' if scenario_b_ok else 'FAIL'}")
    
    if scenario_a_ok and scenario_b_ok:
        results['status'] = 'success'
        print(f"\n🎉 COMPLETE PIPELINE TEST: ✅ SUCCESS")
        
        # Comparison
        if 'analysis' in results['scenario_a'] and 'analysis' in results['scenario_b']:
            a = results['scenario_a']['analysis']
            b = results['scenario_b']['analysis']
            
            print(f"\n📊 Comparison:")
            print(f"   {'Metric':<30} {'Scenario A':<20} {'Scenario B':<20}")
            print(f"   {'-'*70}")
            print(f"   {'AUC':<30} {a['auc']:<20.4f} {b['auc']:<20.4f}")
            print(f"   {'Precision':<30} {a['precision']:<20.4f} {b['precision']:<20.4f}")
            print(f"   {'Recall':<30} {a['recall']:<20.4f} {b['recall']:<20.4f}")
            print(f"   {'F1 Score':<30} {a['f1']:<20.4f} {b['f1']:<20.4f}")
    else:
        results['status'] = 'failed'
        print(f"\n❌ COMPLETE PIPELINE TEST: ❌ FAILED")
    
    # Save results
    output_file = Path('result/pipeline_test_final.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Test results saved to: {output_file}")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return scenario_a_ok and scenario_b_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

