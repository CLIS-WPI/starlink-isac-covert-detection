#!/usr/bin/env python3
"""
Diagnostic Tests for CNN Detection
===================================
Runs multiple tests to diagnose why AUC is low.
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def wait_for_training_completion(process_name, timeout=1800):
    """Wait for training to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = subprocess.run(['pgrep', '-f', process_name], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            return True
        time.sleep(10)
    return False

def check_training_result(log_file, result_file):
    """Check training result from log and JSON."""
    results = {}
    
    # Check log file
    if Path(log_file).exists():
        with open(log_file, 'r') as f:
            content = f.read()
            if 'Test AUC:' in content:
                for line in content.split('\n'):
                    if 'Test AUC:' in line:
                        try:
                            auc = float(line.split('Test AUC:')[1].strip().split()[0])
                            results['auc'] = auc
                        except:
                            pass
    
    # Check JSON result
    if Path(result_file).exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            metrics = data.get('metrics', {})
            results.update({
                'auc': metrics.get('auc', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
            })
    
    return results

def test_with_tx_grids():
    """Test 1: Train with tx_grids (pre-channel)."""
    print_header("Test 1: Training with tx_grids (pre-channel)")
    
    print("Purpose: Diagnose if problem is from channel or CNN")
    print("Expected: If AUC is high → problem is from channel")
    print("          If AUC is still low → problem is from CNN architecture\n")
    
    # Training should already be running
    print("⏳ Waiting for training to complete...")
    
    if wait_for_training_completion("main_detection_cnn.py.*ground", timeout=1800):
        print("✅ Training completed!")
        
        result_file = Path('result/scenario_b/detection_results_cnn.json')
        log_file = Path('training_scenario_b_tx_grids.log')
        
        results = check_training_result(log_file, result_file)
        
        if results:
            print(f"\n📊 Results:")
            print(f"   AUC: {results.get('auc', 0):.4f}")
            print(f"   Precision: {results.get('precision', 0):.4f}")
            print(f"   Recall: {results.get('recall', 0):.4f}")
            print(f"   F1: {results.get('f1', 0):.4f}")
            
            if results.get('auc', 0) > 0.8:
                print("\n✅ Conclusion: Problem is from channel/equalization!")
                print("   → Need to improve equalization or use CSI fusion")
                return 'channel_problem', results
            else:
                print("\n⚠️  Conclusion: Problem might be from CNN architecture")
                print("   → Need to try CSI fusion or improve CNN")
                return 'cnn_problem', results
        else:
            print("⚠️  Could not extract results")
            return None, None
    else:
        print("❌ Training timeout")
        return None, None

def test_with_csi_fusion():
    """Test 2: Train with CSI fusion."""
    print_header("Test 2: Training with CSI fusion (--use-csi)")
    
    print("Purpose: Improve detection using multi-modal fusion")
    print("Expected: AUC should improve with CSI information\n")
    
    cmd = [
        'python3', 'main_detection_cnn.py',
        '--scenario', 'ground',
        '--epochs', '30',
        '--batch-size', '512',
        '--use-csi'
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Starting training...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ Training completed in {elapsed/60:.1f} minutes")
        
        result_file = Path('result/scenario_b/detection_results_cnn_csi.json')
        results = check_training_result(None, result_file)
        
        if results:
            print(f"\n📊 Results:")
            print(f"   AUC: {results.get('auc', 0):.4f}")
            print(f"   Precision: {results.get('precision', 0):.4f}")
            print(f"   Recall: {results.get('recall', 0):.4f}")
            print(f"   F1: {results.get('f1', 0):.4f}")
            
            return results
    else:
        print(f"❌ Training failed: {result.stderr[:500]}")
        return None
    
    return None

def generate_final_report(test1_result, test2_result):
    """Generate final diagnostic report."""
    print_header("📋 Final Diagnostic Report")
    
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("="*80)
    print("TEST 1: tx_grids (pre-channel)")
    print("="*80)
    if test1_result:
        print(f"✅ Completed")
        print(f"   AUC: {test1_result.get('auc', 0):.4f}")
        print(f"   Conclusion: {'Channel problem' if test1_result.get('auc', 0) > 0.8 else 'CNN problem'}")
    else:
        print("❌ Not completed")
    
    print("\n" + "="*80)
    print("TEST 2: CSI Fusion")
    print("="*80)
    if test2_result:
        print(f"✅ Completed")
        print(f"   AUC: {test2_result.get('auc', 0):.4f}")
        print(f"   Improvement: {test2_result.get('auc', 0) - (test1_result.get('auc', 0) if test1_result else 0.53):.4f}")
    else:
        print("⏳ Not completed yet")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if test1_result and test1_result.get('auc', 0) > 0.8:
        print("✅ Use tx_grids for training (pre-channel)")
        print("   → Pattern is visible in pre-channel signal")
        print("   → Problem is from channel/equalization")
    elif test2_result and test2_result.get('auc', 0) > 0.7:
        print("✅ Use CSI fusion (--use-csi)")
        print("   → CSI fusion improves detection")
    else:
        print("⚠️  Need to improve:")
        print("   1. CNN architecture")
        print("   2. Preprocessing")
        print("   3. Feature extraction")
    
    print("\n" + "="*80)

def main():
    """Main execution."""
    print_header("🔍 Diagnostic Tests for CNN Detection")
    
    # Test 1: tx_grids
    test1_status, test1_result = test_with_tx_grids()
    
    # Test 2: CSI fusion
    test2_result = None
    if test1_status:
        test2_result = test_with_csi_fusion()
    
    # Generate report
    generate_final_report(test1_result, test2_result)

if __name__ == "__main__":
    main()

