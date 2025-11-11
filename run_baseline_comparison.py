#!/usr/bin/env python3
"""
Run baseline comparison for both scenarios
"""
import subprocess
import sys
from pathlib import Path


def run_baseline(scenario='sat'):
    """Run baseline detection for a scenario."""
    scenario_name = 'A' if scenario == 'sat' else 'B'
    
    print("\n" + "="*80)
    print(f"🎯 Running Baseline Comparison for Scenario {scenario_name}")
    print("="*80)
    
    cmd = f"python3 baseline_detection.py --scenario {scenario} --svm-features 50"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✅ Baseline comparison for Scenario {scenario_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Baseline comparison for Scenario {scenario_name} failed: {e}")
        return False


def main():
    """Run baseline comparison for all scenarios."""
    print("="*80)
    print("🎯 Baseline Comparison - All Scenarios")
    print("="*80)
    
    scenarios = [
        ('sat', 'A'),
        ('ground', 'B')
    ]
    
    results = {}
    
    for scenario, name in scenarios:
        # Check if dataset exists
        dataset_file = Path(f'dataset/dataset_scenario_{"a" if scenario == "sat" else "b"}_5000.pkl')
        if not dataset_file.exists():
            print(f"\n⚠️  Scenario {name}: Dataset not found, skipping...")
            results[name] = 'skipped'
            continue
        
        # Run baseline
        success = run_baseline(scenario)
        results[name] = 'success' if success else 'failed'
    
    # Summary
    print("\n" + "="*80)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("="*80)
    
    for scenario, status in results.items():
        emoji = '✅' if status == 'success' else '⚠️' if status == 'skipped' else '❌'
        print(f"   {emoji} Scenario {scenario}: {status.upper()}")
    
    print("="*80)


if __name__ == "__main__":
    main()

