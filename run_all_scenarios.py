#!/usr/bin/env python3
"""
🚀 Automated Pipeline for Scenario A and B
==========================================
This script automatically executes the complete pipeline for both scenarios:
- Scenario A: Insider@Satellite (Downlink)
- Scenario B: Insider@Ground (Uplink → Relay → Downlink)

For each scenario:
1. Generate dataset
2. Train CNN-only model
3. Train CNN+CSI model
4. Save datasets with appropriate names

Usage:
    python3 run_all_scenarios.py
    python3 run_all_scenarios.py --num-samples 500
    python3 run_all_scenarios.py --skip-scenario-b  # Only run Scenario A
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

# Configuration
from config.settings import (
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR,
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA
)


def run_command(cmd, description, check=True):
    """
    Run a shell command and handle errors.
    
    Args:
        cmd: Command to run (list of strings)
        description: Description of what the command does
        check: If True, exit on error
    """
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=check)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed with exit code {result.returncode}")
        if check:
            sys.exit(1)
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def update_settings_file(setting_key, setting_value, file_path="config/settings.py"):
    """
    Update a setting in config/settings.py.
    
    Args:
        setting_key: Name of the setting (e.g., 'INSIDER_MODE')
        setting_value: New value (e.g., "'sat'" or "500")
        file_path: Path to settings file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the setting
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(setting_key + ' ='):
            # Preserve indentation
            indent = len(line) - len(line.lstrip())
            
            # Extract comment if exists (everything after # on the same line)
            comment = ''
            if '#' in line:
                # Find the # symbol and everything after it
                hash_idx = line.find('#')
                comment = line[hash_idx:].rstrip()
            
            # Build new line: preserve indent, set value, preserve comment
            if comment:
                lines[i] = ' ' * indent + f"{setting_key} = {setting_value}  {comment}\n"
            else:
                lines[i] = ' ' * indent + f"{setting_key} = {setting_value}\n"
            
            updated = True
            break
    
    if updated:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"  ✓ Updated {setting_key} = {setting_value}")
    else:
        print(f"  ⚠️  Warning: {setting_key} not found in {file_path}")


def run_scenario(scenario_name, insider_mode, num_samples, num_satellites):
    """
    Run complete pipeline for one scenario.
    
    Args:
        scenario_name: 'A' or 'B'
        insider_mode: 'sat' or 'ground'
        num_samples: Number of samples per class
        num_satellites: Number of satellites
    """
    print(f"\n{'#'*70}")
    print(f"# SCENARIO {scenario_name} - Insider@{'Satellite' if insider_mode == 'sat' else 'Ground'}")
    print(f"{'#'*70}")
    
    # Step 1: Update settings
    print(f"\n📝 Step 1: Updating settings for Scenario {scenario_name}...")
    update_settings_file('INSIDER_MODE', f"'{insider_mode}'")
    update_settings_file('NUM_SAMPLES_PER_CLASS', str(num_samples))
    update_settings_file('NUM_SATELLITES_FOR_TDOA', str(num_satellites))
    
    # Step 2: Generate dataset
    print(f"\n📊 Step 2: Generating dataset for Scenario {scenario_name}...")
    dataset_cmd = ['python3', 'generate_dataset_parallel.py']
    run_command(dataset_cmd, f"Generate dataset for Scenario {scenario_name}")
    
    # Step 3: Rename dataset to scenario-specific name
    default_dataset = f"{DATASET_DIR}/dataset_samples{num_samples}_sats{num_satellites}.pkl"
    scenario_dataset = f"{DATASET_DIR}/dataset_scenario_{scenario_name.lower()}.pkl"
    
    if os.path.exists(default_dataset):
        if os.path.exists(scenario_dataset):
            os.remove(scenario_dataset)  # Remove old dataset if exists
        shutil.move(default_dataset, scenario_dataset)
        print(f"  ✓ Dataset saved as: {scenario_dataset}")
    else:
        print(f"  ⚠️  Warning: Dataset not found at {default_dataset}")
    
    # Step 4: Train CNN-only
    print(f"\n🧠 Step 4: Training CNN-only for Scenario {scenario_name}...")
    cnn_only_cmd = [
        'python3', 'main_detection_cnn.py',
        '--epochs', '50',
        '--batch-size', '512'
    ]
    run_command(cnn_only_cmd, f"Train CNN-only for Scenario {scenario_name}")
    
    # Step 5: Train CNN+CSI
    print(f"\n🧠 Step 5: Training CNN+CSI for Scenario {scenario_name}...")
    cnn_csi_cmd = [
        'python3', 'main_detection_cnn.py',
        '--use-csi',
        '--epochs', '50',
        '--batch-size', '512'
    ]
    run_command(cnn_csi_cmd, f"Train CNN+CSI for Scenario {scenario_name}")
    
    print(f"\n✅ Scenario {scenario_name} completed successfully!")
    print(f"   - Dataset: {scenario_dataset}")
    print(f"   - Results: {RESULT_DIR}/scenario_{scenario_name.lower()}/")
    print(f"   - Models: {MODEL_DIR}/scenario_{scenario_name.lower()}/")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Automated pipeline for Scenario A and B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_all_scenarios.py
  python3 run_all_scenarios.py --num-samples 500
  python3 run_all_scenarios.py --skip-scenario-b
  python3 run_all_scenarios.py --num-samples 1500 --num-satellites 12
        """
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=NUM_SAMPLES_PER_CLASS,
        help=f'Number of samples per class (default: {NUM_SAMPLES_PER_CLASS})'
    )
    
    parser.add_argument(
        '--num-satellites',
        type=int,
        default=NUM_SATELLITES_FOR_TDOA,
        help=f'Number of satellites (default: {NUM_SATELLITES_FOR_TDOA})'
    )
    
    parser.add_argument(
        '--skip-scenario-a',
        action='store_true',
        help='Skip Scenario A (only run Scenario B)'
    )
    
    parser.add_argument(
        '--skip-scenario-b',
        action='store_true',
        help='Skip Scenario B (only run Scenario A)'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print("="*70)
    print("🚀 AUTOMATED PIPELINE FOR SCENARIO A AND B")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Samples per class: {args.num_samples}")
    print(f"  - Total samples: {args.num_samples * 2}")
    print(f"  - Number of satellites: {args.num_satellites}")
    print(f"  - Skip Scenario A: {args.skip_scenario_a}")
    print(f"  - Skip Scenario B: {args.skip_scenario_b}")
    print("="*70)
    
    # Run Scenario A
    if not args.skip_scenario_a:
        run_scenario('A', 'sat', args.num_samples, args.num_satellites)
    else:
        print("\n⏭️  Skipping Scenario A (--skip-scenario-a)")
    
    # Run Scenario B
    if not args.skip_scenario_b:
        run_scenario('B', 'ground', args.num_samples, args.num_satellites)
    else:
        print("\n⏭️  Skipping Scenario B (--skip-scenario-b)")
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n📁 Output Files:")
    print(f"  Datasets:")
    print(f"    - {DATASET_DIR}/dataset_scenario_a.pkl")
    if not args.skip_scenario_b:
        print(f"    - {DATASET_DIR}/dataset_scenario_b.pkl")
    print(f"\n  Results:")
    print(f"    - {RESULT_DIR}/scenario_a/")
    if not args.skip_scenario_b:
        print(f"    - {RESULT_DIR}/scenario_b/")
    print(f"\n  Models:")
    print(f"    - {MODEL_DIR}/scenario_a/")
    if not args.skip_scenario_b:
        print(f"    - {MODEL_DIR}/scenario_b/")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

