#!/usr/bin/env python3
"""
Wait for dataset generation and then analyze
"""
import time
import subprocess
from pathlib import Path

def wait_for_dataset(max_wait_minutes=60):
    """Wait for dataset to be generated."""
    dataset_file = Path('dataset/dataset_scenario_b_5000.pkl')
    start_time = time.time()
    
    print("="*80)
    print("⏳ Waiting for dataset generation...")
    print("="*80)
    
    while True:
        if dataset_file.exists():
            file_size = dataset_file.stat().st_size / (1024**2)  # MB
            if file_size > 100:  # At least 100 MB
                print(f"\n✅ Dataset ready! Size: {file_size:.1f} MB")
                elapsed = (time.time() - start_time) / 60
                print(f"   Time elapsed: {elapsed:.1f} minutes")
                return True
        
        elapsed = (time.time() - start_time) / 60
        if elapsed > max_wait_minutes:
            print(f"\n❌ Timeout after {max_wait_minutes} minutes")
            return False
        
        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"   Still waiting... ({elapsed:.1f} minutes)")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    if wait_for_dataset(60):
        print("\n" + "="*80)
        print("🚀 Starting analysis...")
        print("="*80)
        
        # Run ceiling test
        print("\n1. Running ceiling test with H_true...")
        subprocess.run(['python3', 'test_eq_ceiling_h_true.py'], check=False)
        
        # Generate detailed logs
        print("\n2. Generating detailed logs...")
        subprocess.run(['python3', 'generate_detailed_logs.py'], check=False)
        
        print("\n" + "="*80)
        print("✅ Analysis complete!")
        print("="*80)
    else:
        print("\n❌ Dataset generation timeout")

