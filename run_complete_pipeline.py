#!/usr/bin/env python3
"""
Complete Pipeline: Scenario A and B
===================================
Runs complete pipeline for both scenarios:
1. Dataset generation
2. Model training
3. Analysis
"""
import os
import sys
import subprocess
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status."""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        return True
    else:
        print(f"❌ {description} failed")
        print(f"Error: {result.stderr}")
        return False

def analyze_scenario(scenario_name, scenario_label):
    """Analyze scenario results."""
    print("\n" + "="*80)
    print(f"📊 Analysis: {scenario_label}")
    print("="*80)
    
    # Load dataset
    dataset_file = Path(f'dataset/dataset_{scenario_name}_5000.pkl')
    if not dataset_file.exists():
        print(f"❌ Dataset not found: {dataset_file}")
        return None
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Load training results
    result_file = Path(f'result/{scenario_name}/detection_results_cnn.json')
    if not result_file.exists():
        print(f"❌ Results not found: {result_file}")
        return None
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    metrics = results.get('metrics', {})
    
    # Analyze dataset
    meta = dataset.get('meta', [])
    preservations = []
    snr_improvements = []
    
    for m in meta[:1000]:  # Sample 1000
        if isinstance(m, tuple):
            _, m = m
        if 'eq_pattern_preservation' in m:
            preservations.append(m['eq_pattern_preservation'])
        if 'eq_snr_improvement_db' in m:
            snr_improvements.append(m['eq_snr_improvement_db'])
    
    analysis = {
        'scenario': scenario_label,
        'dataset_size': len(dataset['labels']),
        'dataset_file_size_mb': dataset_file.stat().st_size / (1024**2),
        'auc': metrics.get('auc', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        'pattern_preservation_median': np.median(preservations) if preservations else 0,
        'pattern_preservation_mean': np.mean(preservations) if preservations else 0,
        'snr_improvement_mean': np.mean(snr_improvements) if snr_improvements else 0,
        'snr_improvement_median': np.median(snr_improvements) if snr_improvements else 0,
    }
    
    print(f"\n📊 Results:")
    print(f"   Dataset: {analysis['dataset_size']} samples ({analysis['dataset_file_size_mb']:.2f} MB)")
    print(f"   AUC: {analysis['auc']:.4f}")
    print(f"   Precision: {analysis['precision']:.4f}")
    print(f"   Recall: {analysis['recall']:.4f}")
    print(f"   F1: {analysis['f1']:.4f}")
    
    if preservations:
        print(f"\n📊 EQ Performance:")
        print(f"   Pattern Preservation: {analysis['pattern_preservation_median']:.3f} (median)")
        print(f"   SNR Improvement: {analysis['snr_improvement_mean']:.2f} dB (mean)")
    
    return analysis

def main():
    """Run complete pipeline."""
    print("="*80)
    print("🚀 COMPLETE PIPELINE: Scenario A and B")
    print("="*80)
    
    results = {}
    
    # ===== Scenario A =====
    print("\n" + "="*80)
    print("1️⃣  SCENARIO A (Single-hop Downlink)")
    print("="*80)
    
    # Generate dataset
    if not run_command(
        "python3 generate_dataset_parallel.py --scenario sat --total-samples 5000",
        "Scenario A: Dataset Generation"
    ):
        print("❌ Scenario A dataset generation failed")
        return
    
    # Copy to standard name
    subprocess.run("cp dataset/dataset_scenario_a_4998.pkl dataset/dataset_scenario_a_5000.pkl 2>/dev/null || true", shell=True)
    
    # Train model
    if not run_command(
        "python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512",
        "Scenario A: Model Training"
    ):
        analysis_a = analyze_scenario('scenario_a', 'Scenario A')
        results['scenario_a'] = analysis_a
    
    # ===== Scenario B =====
    print("\n" + "="*80)
    print("2️⃣  SCENARIO B (Dual-hop Relay)")
    print("="*80)
    
    # Generate dataset
    if not run_command(
        "python3 generate_dataset_parallel.py --scenario ground --total-samples 5000",
        "Scenario B: Dataset Generation"
    ):
        print("❌ Scenario B dataset generation failed")
        return
    
    # Copy to standard name
    subprocess.run("cp dataset/dataset_scenario_b_4998.pkl dataset/dataset_scenario_b_5000.pkl 2>/dev/null || true", shell=True)
    
    # Train model
    if not run_command(
        "python3 main_detection_cnn.py --scenario ground --epochs 30 --batch-size 512",
        "Scenario B: Model Training"
    ):
        analysis_b = analyze_scenario('scenario_b', 'Scenario B')
        results['scenario_b'] = analysis_b
    
    # ===== Final Report =====
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON REPORT")
    print("="*80)
    
    if 'scenario_a' in results and 'scenario_b' in results:
        a = results['scenario_a']
        b = results['scenario_b']
        
        print(f"\n{'Metric':<30} {'Scenario A':<20} {'Scenario B':<20}")
        print("-" * 70)
        print(f"{'AUC':<30} {a['auc']:<20.4f} {b['auc']:<20.4f}")
        print(f"{'Precision':<30} {a['precision']:<20.4f} {b['precision']:<20.4f}")
        print(f"{'Recall':<30} {a['recall']:<20.4f} {b['recall']:<20.4f}")
        print(f"{'F1 Score':<30} {a['f1']:<20.4f} {b['f1']:<20.4f}")
        print(f"{'Pattern Preservation':<30} {a['pattern_preservation_median']:<20.3f} {b['pattern_preservation_median']:<20.3f}")
        print(f"{'SNR Improvement (dB)':<30} {a['snr_improvement_mean']:<20.2f} {b['snr_improvement_mean']:<20.2f}")
        
        # Save report
        report_file = Path(f'result/comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Report saved to: {report_file}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("="*80)

if __name__ == "__main__":
    main()

