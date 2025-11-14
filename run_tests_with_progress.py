#!/usr/bin/env python3
"""
🧪 Run All Tests with Progress Bar and GPU Support
===================================================
Executes all tests with real-time progress tracking and GPU utilization.
"""
import subprocess
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed. Install with: pip install tqdm")
    print("   Progress bar will be disabled.\n")


def check_gpu_availability():
    """Check if GPU is available and configure it."""
    try:
        import tensorflow as tf
        
        # Force GPU usage if available
        print("🔍 Checking for GPU devices...")
        
        # List all devices
        print("\n📋 All available devices:")
        all_devices = tf.config.list_physical_devices()
        for device in all_devices:
            print(f"   - {device}")
        
        # Check GPU specifically
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                try:
                    # Enable memory growth to avoid allocating all memory
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"   ✅ GPU {i} memory growth enabled")
                except Exception as e:
                    print(f"   ⚠️  Could not set memory growth for GPU {i}: {e}")
            
            # Force TensorFlow to use GPU
            print("\n🧪 Testing GPU computation...")
            try:
                with tf.device('/GPU:0'):
                    # Create a simple computation
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                    result = tf.reduce_sum(c)
                    print(f"   ✅ GPU computation successful!")
                    print(f"   Result shape: {c.shape}, Sum: {result.numpy():.2f}")
                    print(f"   ✅ TensorFlow is using GPU!")
            except Exception as e:
                print(f"   ❌ GPU computation failed: {e}")
                return False
            
            # Set default device to GPU
            print("\n🚀 Configuring TensorFlow to use GPU by default...")
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print("   ✅ GPU set as default device")
            except Exception as e:
                print(f"   ⚠️  Could not set default device: {e}")
            
            return True
        else:
            print("\n⚠️  No GPU devices found.")
            print("   TensorFlow will use CPU")
            print("   (This is normal if running on a system without GPU)")
            return False
    except Exception as e:
        print(f"\n❌ Could not check GPU: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests_with_progress():
    """Run all tests with progress tracking."""
    print("="*70)
    print("🧪 Test Execution with Progress Tracking")
    print("="*70)
    print()
    
    # Check GPU
    has_gpu = check_gpu_availability()
    print()
    
    # Set environment variables for GPU
    if has_gpu:
        # Don't set CUDA_VISIBLE_DEVICES if GPU is already configured
        # Let TensorFlow use all available GPUs
        print("🚀 Using GPU for acceleration")
        print("   TensorFlow will use GPU automatically")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU
        print("💻 Using CPU")
    print()
    
    # Collect test files first to get total count
    print("📋 Collecting tests...")
    collect_result = subprocess.run(
        [
            'python3', '-m', 'pytest',
            'tests/',
            '--collect-only',
            '-q'
        ],
        capture_output=True,
        text=True
    )
    
    # Parse collected tests - look for "collected X items" line
    total_tests = 0
    for line in collect_result.stdout.split('\n'):
        if 'collected' in line.lower() and 'item' in line.lower():
            # Extract number from "collected X items"
            import re
            match = re.search(r'collected\s+(\d+)\s+item', line.lower())
            if match:
                total_tests = int(match.group(1))
                break
    
    if total_tests == 0:
        # Fallback: count test lines
        test_lines = [line for line in collect_result.stdout.split('\n') if '::' in line and 'test_' in line]
        total_tests = len(test_lines)
    
    print(f"✅ Found {total_tests} tests to run")
    print()
    
    # Run tests with JSON report
    print("="*70)
    print("🚀 Running Tests...")
    print("="*70)
    print()
    
    if HAS_TQDM:
        # Use tqdm for progress bar
        with tqdm(total=total_tests, desc="Tests", unit="test", ncols=100, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Run pytest with live output
            process = subprocess.Popen(
                [
                    'python3', '-m', 'pytest',
                    'tests/',
                    '-v',
                    '--json-report',
                    '--json-report-file=test_results.json',
                    '--tb=short',
                    '--maxfail=10',  # Allow more failures before stopping
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            passed = 0
            failed = 0
            skipped = 0
            current_test = ""
            
            # Read output line by line
            for line in process.stdout:
                # Print to console
                sys.stdout.write(line)
                sys.stdout.flush()
                
                # Update progress bar
                if 'PASSED' in line:
                    passed += 1
                    pbar.update(1)
                    pbar.set_postfix({'✅': passed, '❌': failed, '⏭️': skipped})
                elif 'FAILED' in line:
                    failed += 1
                    pbar.update(1)
                    pbar.set_postfix({'✅': passed, '❌': failed, '⏭️': skipped})
                elif 'SKIPPED' in line:
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix({'✅': passed, '❌': failed, '⏭️': skipped})
            
            process.wait()
            returncode = process.returncode
            
    else:
        # Fallback: run without progress bar
        result = subprocess.run(
            [
                'python3', '-m', 'pytest',
                'tests/',
                '-v',
                '--json-report',
                '--json-report-file=test_results.json',
                '--tb=short',
                '--maxfail=10',
            ],
            text=True
        )
        returncode = result.returncode
    
    print()
    print("="*70)
    return returncode


def parse_json_report(json_file='test_results.json'):
    """Parse JSON report and extract statistics."""
    if not Path(json_file).exists():
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract summary
    summary = data.get('summary', {})
    total = summary.get('total', 0)
    passed = summary.get('passed', 0)
    failed = summary.get('failed', 0)
    skipped = summary.get('skipped', 0)
    error = summary.get('error', 0)
    duration = summary.get('duration', 0)
    
    # Extract test results by file
    tests_by_file = defaultdict(lambda: {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0})
    failed_tests = []
    
    for test in data.get('tests', []):
        nodeid = test.get('nodeid', '')
        outcome = test.get('outcome', 'unknown')
        
        # Extract file name
        if '::' in nodeid:
            file_path = nodeid.split('::')[0]
            file_name = Path(file_path).name
        else:
            file_name = 'unknown'
        
        tests_by_file[file_name][outcome] = tests_by_file[file_name][outcome] + 1
        
        if outcome == 'failed':
            failed_tests.append({
                'nodeid': nodeid,
                'call': test.get('call', {}),
                'setup': test.get('setup', {}),
                'teardown': test.get('teardown', {})
            })
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'error': error,
        'duration': duration,
        'tests_by_file': dict(tests_by_file),
        'failed_tests': failed_tests
    }


def generate_markdown_report(stats, output_file='TEST_EXECUTION_REPORT.md'):
    """Generate comprehensive Markdown report."""
    print("="*70)
    print("📝 Generating Markdown Report...")
    print("="*70)
    
    report = []
    report.append("# 🧪 Test Execution Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    report.append(f"**Duration:** {stats['duration']:.2f} seconds\n")
    report.append("\n---\n")
    
    # Summary
    report.append("## 📊 Summary\n\n")
    report.append("| Metric | Count | Percentage |\n")
    report.append("|--------|-------|------------|\n")
    report.append(f"| **Total Tests** | {stats['total']} | 100% |\n")
    report.append(f"| ✅ **Passed** | {stats['passed']} | {stats['passed']/stats['total']*100:.1f}% |\n")
    report.append(f"| ❌ **Failed** | {stats['failed']} | {stats['failed']/stats['total']*100:.1f}% |\n")
    report.append(f"| ⏭️  **Skipped** | {stats['skipped']} | {stats['skipped']/stats['total']*100:.1f}% |\n")
    report.append(f"| ⚠️  **Error** | {stats['error']} | {stats['error']/stats['total']*100:.1f}% |\n")
    report.append("\n")
    
    status = "✅ ALL PASSED" if stats['failed'] == 0 and stats['error'] == 0 else "❌ SOME FAILED"
    report.append(f"**Status:** {status}\n")
    report.append("\n---\n")
    
    # Results by file
    report.append("## 📄 Results by File\n\n")
    for file_name, counts in sorted(stats['tests_by_file'].items()):
        total_file = sum(counts.values())
        status_icon = "✅" if counts['failed'] == 0 and counts['error'] == 0 else "❌"
        report.append(f"### {status_icon} `{file_name}`\n\n")
        report.append(f"- **Total:** {total_file} | **Passed:** {counts['passed']} | **Failed:** {counts['failed']} | **Skipped:** {counts['skipped']} | **Error:** {counts['error']}\n\n")
    
    report.append("---\n")
    
    # Failed tests
    if stats['failed_tests']:
        report.append("## ❌ Failed Tests Details\n\n")
        for i, test in enumerate(stats['failed_tests'][:10], 1):  # Limit to 10
            report.append(f"### `{test['nodeid']}`\n\n")
            if 'longrepr' in test.get('call', {}):
                report.append("```\n")
                report.append(test['call']['longrepr'])
                report.append("\n```\n\n")
    
    report.append("---\n")
    
    # Notes
    report.append("## 📝 Notes\n\n")
    report.append("- Tests marked with `@pytest.mark.slow` may take several minutes\n")
    report.append("- Some tests require dataset files in `dataset/` directory\n")
    report.append("- Some tests require result files in `result/` directory\n")
    report.append("- Skipped tests are expected if prerequisites are missing\n")
    report.append("\n---\n")
    report.append(f"\n**Report generated by:** `run_tests_with_progress.py`  \n")
    report.append(f"**JSON Report:** `test_results.json`\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write(''.join(report))
    
    print(f"✅ Markdown report generated: {output_file}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("🧪 Test Execution and Report Generation")
    print("="*70 + "\n")
    
    # Run tests
    returncode = run_tests_with_progress()
    
    # Parse results
    print("\n" + "="*70)
    print("📊 Parsing Results...")
    print("="*70)
    
    stats = parse_json_report()
    
    if stats:
        print(f"\n✅ Total: {stats['total']}")
        print(f"✅ Passed: {stats['passed']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"⏭️  Skipped: {stats['skipped']}")
        print(f"⚠️  Error: {stats['error']}")
        print(f"⏱️  Duration: {stats['duration']:.2f}s")
        
        # Generate report
        print("\n" + "="*70)
        generate_markdown_report(stats)
        
        print("\n" + "="*70)
        print("✅ Done!")
        print("="*70)
        print("\n📄 Reports generated:")
        print("   - test_results.json (JSON)")
        print("   - TEST_EXECUTION_REPORT.md (Markdown)")
        print()
    else:
        print("❌ Could not parse test results")
        return 1
    
    return 0 if returncode == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

