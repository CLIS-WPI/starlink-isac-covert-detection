#!/usr/bin/env python3
"""
Scenario B: Complete Pipeline
==============================
Generates dataset, trains model, and produces results for Scenario B.
Scenario B: Dual-hop Uplink-Relay-Downlink (Insider@Ground)
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

def verify_dataset(expected_samples=10000):
    """Verify dataset exists and is valid."""
    dataset_file = Path(f'dataset/dataset_scenario_b_{expected_samples}.pkl')
    
    if not dataset_file.exists():
        # Try to find any file close to expected (tolerance ±10 samples)
        import glob
        pattern = 'dataset/dataset_scenario_b_*.pkl'
        candidates = glob.glob(pattern)
        
        # Find closest match
        best_match = None
        min_diff = float('inf')
        
        for candidate in candidates:
            # Extract sample count from filename
            import re
            match = re.search(r'dataset_scenario_b_(\d+)\.pkl', candidate)
            if match:
                sample_count = int(match.group(1))
                diff = abs(sample_count - expected_samples)
                if diff < min_diff and diff <= 10:  # Allow ±10 tolerance
                    min_diff = diff
                    best_match = candidate
        
        if best_match:
            import shutil
            fallback_file = Path(best_match)
            shutil.copy2(fallback_file, dataset_file)
            print(f"✅ Using {fallback_file.name} (close to {expected_samples} samples)")
            
            # Clean up old file
            try:
                fallback_file.unlink()
            except:
                pass
        else:
            return False, "Dataset file not found"
    
    try:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        
        actual_samples = len(dataset.get('labels', []))
        # Allow ±10 samples tolerance for multiprocessing variations
        if abs(actual_samples - expected_samples) > 10:
            return False, f"Expected ~{expected_samples} samples, got {actual_samples}"
        
        if 'rx_grids' not in dataset:
            return False, "Missing required field (rx_grids)"
        
        # 🔧 IMPROVED: Fail-fast check for EQ metadata completeness
        # Verify that EQ metadata exists for most samples (>=80%)
        if 'meta' in dataset and len(dataset['meta']) > 0:
            eq_metadata_count = 0
            pattern_preservation_count = 0
            total_meta = len(dataset['meta'])
            
            for meta in dataset['meta']:
                if isinstance(meta, dict):
                    if 'eq_snr_improvement_db' in meta:
                        eq_metadata_count += 1
                    if 'eq_pattern_preservation' in meta:
                        pattern_preservation_count += 1
            
            eq_coverage = eq_metadata_count / total_meta if total_meta > 0 else 0
            preservation_coverage = pattern_preservation_count / total_meta if total_meta > 0 else 0
            
            if eq_coverage < 0.80:
                return False, f"Insufficient EQ metadata: only {eq_coverage*100:.1f}% samples have eq_snr_improvement_db (required >=80%)"
            
            if preservation_coverage < 0.80:
                print(f"   ⚠️  Warning: Only {preservation_coverage*100:.1f}% samples have eq_pattern_preservation")
            
            # Show sample metadata
            sample_meta = dataset['meta'][0]
            if 'eq_snr_improvement_db' in sample_meta:
                snr_improvement = sample_meta.get('eq_snr_improvement_db', 0)
                if snr_improvement > 0:
                    print(f"  ✓ Dataset contains equalized grids (SNR improvement: {snr_improvement:.2f} dB)")
                    print(f"  ✓ EQ metadata coverage: {eq_coverage*100:.1f}%")
        
        size_mb = dataset_file.stat().st_size / (1024**2)
        return True, f"Dataset valid: {len(dataset['labels'])} samples, {size_mb:.2f} MB"
    except Exception as e:
        return False, f"Error loading dataset: {e}"

def verify_training_results():
    """Verify training results exist."""
    result_file = Path('result/scenario_b/detection_results_cnn.json')
    
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
    print("📊 SCENARIO B - FINAL RESULTS")
    print("="*80)
    
    # Load dataset (try 10K first, fallback to 5K)
    dataset_file = Path('dataset/dataset_scenario_b_10000.pkl')
    if not dataset_file.exists():
        dataset_file = Path('dataset/dataset_scenario_b_5000.pkl')
        if not dataset_file.exists():
            print("❌ Dataset file not found!")
            return
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Load results
    result_file = Path('result/scenario_b/detection_results_cnn.json')
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
    
    # Check EQ performance from metadata
    if 'meta' in dataset and len(dataset['meta']) > 0:
        eq_improvements = []
        pattern_preservations = []
        for meta in dataset['meta'][:100]:  # Check first 100 samples
            if isinstance(meta, dict):
                if 'eq_snr_improvement_db' in meta:
                    eq_improvements.append(meta['eq_snr_improvement_db'])
                if 'eq_pattern_preservation' in meta:
                    pattern_preservations.append(meta['eq_pattern_preservation'])
        
        if eq_improvements:
            mean_snr_improvement = np.mean(eq_improvements)
            print(f"\n📊 Equalization Performance:")
            print(f"   • Mean SNR Improvement: {mean_snr_improvement:.2f} dB")
        
        if pattern_preservations:
            median_preservation = np.median(pattern_preservations)
            print(f"   • Median Pattern Preservation: {median_preservation:.4f}")
    
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
    print(f"   • Model: model/scenario_b/cnn_detector.keras")
    print(f"   • Results: result/scenario_b/detection_results_cnn.json")
    print(f"   • Normalization: model/scenario_b/cnn_detector_norm.pkl")
    print(f"   • Training log: training_scenario_b.log")
    
    print("\n" + "="*80)
    print("✅ SCENARIO B PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

def main():
    """Run complete Scenario B pipeline."""
    print("="*80)
    print("🚀 SCENARIO B: Complete Pipeline")
    print("="*80)
    print("Scenario: Dual-hop Uplink-Relay-Downlink (Insider@Ground)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Generate Dataset
    # 🔧 IMPROVED: Timeout can be overridden via environment variable
    timeout_gen = int(os.getenv('TIMEOUT_DATASET_GEN', 5400))  # Default 5400s for 10K samples (longer for EQ)
    success, output = run_command(
        "python3 generate_dataset_parallel.py --scenario ground --total-samples 10000",
        "Step 1: Dataset Generation",
        timeout=timeout_gen,  # Longer timeout for Scenario B (EQ processing)
        log_file=Path('logs') / 'scenario_b_step1_dataset_generation.log'
    )
    
    if not success:
        print("❌ Dataset generation failed!")
        sys.exit(1)
    
    # Verify dataset
    valid, msg = verify_dataset(10000)
    print(f"  Dataset verification: {'✅' if valid else '❌'} {msg}")
    
    if not valid:
        print("❌ Dataset verification failed!")
        sys.exit(1)
    
    # Step 2: Train Model
    # 🔧 IMPROVED: Timeout can be overridden via environment variable
    timeout_train = int(os.getenv('TIMEOUT_MODEL_TRAIN', 3600))  # Default 3600s, can override via env
    success, output = run_command(
        "python3 main_detection_cnn.py --scenario ground --epochs 100 --batch-size 512",
        "Step 2: Model Training",
        timeout=timeout_train,  # Increased timeout for 100 epochs
        log_file=Path('logs') / 'scenario_b_step2_model_training.log'
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
    
    # Step 3: Baseline Comparison
    success, output = run_command(
        "python3 baseline_detection.py --scenario ground --svm-features 50",
        "Step 3: Baseline Comparison",
        timeout=1800,
        log_file=Path('logs') / 'scenario_b_step3_baseline_comparison.log'
    )
    
    if not success:
        print("⚠️  Baseline comparison failed (non-critical)")
    
    # Step 4: Generate Report
    generate_final_report()
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()

