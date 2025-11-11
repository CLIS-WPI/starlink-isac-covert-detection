#!/usr/bin/env python3
"""
Scenario A: Complete Pipeline
==============================
Generates dataset, trains model, and produces results for Scenario A.
Scenario A: Single-hop Downlink (Insider@Satellite)
"""
import os
import sys
import subprocess
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, timeout=1800, log_file=None):
    """Run a command and return success status."""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"Command: {cmd}")
    if log_file:
        print(f"Log file: {log_file}")
    print()
    
    # 🔧 IMPROVED: Save logs to file for debugging and reproducibility
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    if log_file is None:
        # Auto-generate log filename from description
        log_name = description.lower().replace(' ', '_').replace(':', '').replace('step_', 'step')
        log_file = log_dir / f"{log_name}.log"
    
    try:
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            
            # Write to log file
            log_f.write(f"Command: {cmd}\n")
            log_f.write(f"Return code: {result.returncode}\n")
            log_f.write(f"\n{'='*80}\nSTDOUT:\n{'='*80}\n")
            log_f.write(result.stdout)
            log_f.write(f"\n{'='*80}\nSTDERR:\n{'='*80}\n")
            log_f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            print(f"   Log saved to: {log_file}")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr[:500]}")
            print(f"   Full log: {log_file}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"\nTIMEOUT after {timeout}s\n")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} error: {e}")
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"\nEXCEPTION: {e}\n")
        return False, str(e)

def verify_dataset(expected_samples=5000):
    """Verify dataset exists and is valid."""
    dataset_file = Path('dataset/dataset_scenario_a_5000.pkl')
    
    if not dataset_file.exists():
        # Try fallback
        fallback_file = Path('dataset/dataset_scenario_a_4998.pkl')
        if fallback_file.exists():
            import shutil
            shutil.copy2(fallback_file, dataset_file)
            # 🔧 IMPROVED: Verify sample count after copy
            try:
                with open(dataset_file, 'rb') as f:
                    temp_dataset = pickle.load(f)
                actual_samples = len(temp_dataset.get('labels', []))
                if actual_samples != expected_samples:
                    return False, f"Fallback file has {actual_samples} samples, expected {expected_samples}"
            except Exception as e:
                return False, f"Error verifying fallback file: {e}"
            
            # Clean up old file
            try:
                fallback_file.unlink()
                print(f"✅ Copied {fallback_file.name} to {dataset_file.name} and removed old file")
            except:
                print(f"✅ Copied {fallback_file.name} to {dataset_file.name}")
        else:
            return False, "Dataset file not found"
    
    try:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        
        if len(dataset.get('labels', [])) != expected_samples:
            return False, f"Expected {expected_samples} samples, got {len(dataset.get('labels', []))}"
        
        # 🔧 IMPROVED: More lenient check - at least rx_grids is required
        # tx_grids is optional (may not be saved in some configurations)
        if 'rx_grids' not in dataset:
            return False, "Missing required field (rx_grids)"
        if 'tx_grids' not in dataset:
            print("   ⚠️  Warning: tx_grids not found (optional for some configs)")
        
        size_mb = dataset_file.stat().st_size / (1024**2)
        return True, f"Dataset valid: {len(dataset['labels'])} samples, {size_mb:.2f} MB"
    except Exception as e:
        return False, f"Error loading dataset: {e}"

def verify_training_results():
    """Verify training results exist."""
    result_file = Path('result/scenario_a/detection_results_cnn.json')
    
    if not result_file.exists():
        return False, "Results file not found"
    
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

def generate_final_report():
    """Generate final results report."""
    print("\n" + "="*80)
    print("📊 SCENARIO A - FINAL RESULTS")
    print("="*80)
    
    # Load dataset
    dataset_file = Path('dataset/dataset_scenario_a_5000.pkl')
    if not dataset_file.exists():
        print("❌ Dataset file not found!")
        return
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Load results
    result_file = Path('result/scenario_a/detection_results_cnn.json')
    if not result_file.exists():
        print("❌ Results file not found!")
        return
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    metrics = results.get('metrics', {})
    power_analysis = results.get('power_analysis', {})
    
    print(f"\n✅ Dataset Information:")
    print(f"   • Total samples: {len(dataset['labels'])}")
    print(f"   • Dataset size: {dataset_file.stat().st_size / (1024**2):.2f} MB")
    # 🔧 IMPROVED: Robust label handling (works with both list and array)
    labels = np.array(dataset['labels'])
    benign_count = int((labels == 0).sum())
    attack_count = int((labels == 1).sum())
    print(f"   • Benign samples: {benign_count}")
    print(f"   • Attack samples: {attack_count}")
    
    print(f"\n📊 Detection Performance:")
    print(f"   • AUC: {metrics.get('auc', 0):.4f}")
    
    # 🔧 IMPROVED: Report dual thresholds (F1-optimal and Precision-oriented)
    if 'dual_thresholds' in metrics:
        dual = metrics['dual_thresholds']
        f1_opt = dual.get('f1_optimal', {})
        prec_opt = dual.get('precision_oriented', {})
        
        print(f"\n   📌 F1-Optimal Threshold:")
        print(f"      • Threshold: {f1_opt.get('threshold', 0):.4f}")
        print(f"      • Precision: {f1_opt.get('precision', 0):.4f}")
        print(f"      • Recall: {f1_opt.get('recall', 0):.4f}")
        print(f"      • F1 Score: {f1_opt.get('f1', 0):.4f}")
        
        # Only show precision-oriented if it differs from F1-optimal
        if prec_opt and abs(prec_opt.get('threshold', 0) - f1_opt.get('threshold', 0)) > 0.0001:
            print(f"\n   📌 Precision-Oriented Threshold (Precision ≥ 0.70):")
            print(f"      • Threshold: {prec_opt.get('threshold', 0):.4f}")
            print(f"      • Precision: {prec_opt.get('precision', 0):.4f}")
            print(f"      • Recall: {prec_opt.get('recall', 0):.4f}")
            print(f"      • F1 Score: {prec_opt.get('f1', 0):.4f}")
    else:
        # Fallback to single threshold
        print(f"   • Precision: {metrics.get('precision', 0):.4f}")
        print(f"   • Recall: {metrics.get('recall', 0):.4f}")
        print(f"   • F1 Score: {metrics.get('f1', 0):.4f}")
        print(f"   • Optimal Threshold: {metrics.get('threshold', 0):.4f}")
    
    print(f"\n📊 Power Analysis:")
    power_diff = power_analysis.get('difference_pct', 0)
    status = '✅ Ultra-covert' if power_diff < 0.2 else '⚠️  Visible' if power_diff < 1.0 else '❌ Detectable'
    print(f"   • Power Difference: {power_diff:.4f}%")
    print(f"   • Status: {status}")
    
    print(f"\n📁 Output Files:")
    print(f"   • Model: model/scenario_a/cnn_detector.keras")
    print(f"   • Results: result/scenario_a/detection_results_cnn.json")
    print(f"   • Normalization: model/scenario_a/cnn_detector_norm.pkl")
    print(f"   • Training log: training_scenario_a.log")
    
    print("\n" + "="*80)
    print("✅ SCENARIO A PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

def main():
    """Run complete Scenario A pipeline."""
    print("="*80)
    print("🚀 SCENARIO A: Complete Pipeline")
    print("="*80)
    print("Scenario: Single-hop Downlink (Insider@Satellite)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Generate Dataset
    # 🔧 IMPROVED: Timeout can be overridden via environment variable
    timeout_gen = int(os.getenv('TIMEOUT_DATASET_GEN', 1800))  # Default 1800s, can override via env
    success, output = run_command(
        "python3 generate_dataset_parallel.py --scenario sat --total-samples 5000",
        "Step 1: Dataset Generation",
        timeout=timeout_gen,
        log_file=Path('logs') / 'scenario_a_step1_dataset_generation.log'
    )
    
    if not success:
        print("❌ Dataset generation failed!")
        sys.exit(1)
    
    # Verify dataset
    valid, msg = verify_dataset(5000)
    print(f"  Dataset verification: {'✅' if valid else '❌'} {msg}")
    
    if not valid:
        print("❌ Dataset verification failed!")
        sys.exit(1)
    
    # Step 2: Train Model
    # 🔧 IMPROVED: Timeout can be overridden via environment variable
    timeout_train = int(os.getenv('TIMEOUT_MODEL_TRAIN', 3600))  # Default 3600s, can override via env
    success, output = run_command(
        "python3 main_detection_cnn.py --scenario sat --epochs 100 --batch-size 512",
        "Step 2: Model Training",
        timeout=timeout_train,  # Increased timeout for 100 epochs
        log_file=Path('logs') / 'scenario_a_step2_model_training.log'
    )
    
    if not success:
        print("❌ Training failed!")
        sys.exit(1)
    
    # Verify results
    valid, msg = verify_training_results()
    print(f"  Results verification: {'✅' if valid else '❌'} {msg}")
    
    if not valid:
        print("❌ Results verification failed!")
        sys.exit(1)
    
    # Step 3: Generate Report
    generate_final_report()
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()

