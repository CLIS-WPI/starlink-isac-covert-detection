#!/usr/bin/env python3
"""
🧪 Run All Tests and Generate Comprehensive Report
====================================================
Executes all tests and generates both JSON and Markdown reports.
"""
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def run_tests():
    """Run all tests and generate JSON report."""
    print("="*70)
    print("🧪 Running All Tests...")
    print("="*70)
    
    # Run pytest with JSON report
    result = subprocess.run(
        [
            'python3', '-m', 'pytest',
            'tests/',
            '-v',
            '--json-report',
            '--json-report-file=test_results.json',
            '--tb=short',
            '--maxfail=5',  # Stop after 5 failures
        ],
        capture_output=True,
        text=True
    )
    
    # Print output to console
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode, result.stdout


def parse_json_report(json_file='test_results.json'):
    """Parse JSON report and extract statistics."""
    if not Path(json_file).exists():
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def generate_markdown_report(json_data, output_file='TEST_EXECUTION_REPORT.md'):
    """Generate Markdown report from JSON data."""
    
    if json_data is None:
        report = """# 🧪 Test Execution Report

**Status:** ❌ No test results found

Please run tests first:
```bash
python3 run_all_tests_with_report.py
```
"""
        with open(output_file, 'w') as f:
            f.write(report)
        return
    
    # Extract statistics
    summary = json_data.get('summary', {})
    tests = json_data.get('tests', [])
    
    total = summary.get('total', 0)
    passed = summary.get('passed', 0)
    failed = summary.get('failed', 0)
    skipped = summary.get('skipped', 0)
    error = summary.get('error', 0)
    duration = summary.get('duration', 0)
    
    # Group by category
    by_category = defaultdict(lambda: {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0, 'total': 0})
    by_file = defaultdict(lambda: {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0, 'total': 0})
    
    for test in tests:
        nodeid = test.get('nodeid', '')
        outcome = test.get('outcome', 'unknown')
        
        # Extract category (unit/integration/e2e)
        if '/unit/' in nodeid:
            category = 'Unit Tests'
        elif '/integration/' in nodeid:
            category = 'Integration Tests'
        elif '/e2e/' in nodeid:
            category = 'End-to-End Tests'
        else:
            category = 'Other'
        
        # Extract file name
        file_name = nodeid.split('::')[0] if '::' in nodeid else nodeid
        file_name = Path(file_name).name
        
        by_category[category][outcome] = by_category[category].get(outcome, 0) + 1
        by_category[category]['total'] += 1
        
        by_file[file_name][outcome] = by_file[file_name].get(outcome, 0) + 1
        by_file[file_name]['total'] += 1
    
    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# 🧪 Test Execution Report

**Generated:** {timestamp}  
**Duration:** {duration:.2f} seconds

---

## 📊 Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Tests** | {total} | 100% |
| ✅ **Passed** | {passed} | {passed/total*100:.1f}% |
| ❌ **Failed** | {failed} | {failed/total*100:.1f}% |
| ⏭️  **Skipped** | {skipped} | {skipped/total*100:.1f}% |
| ⚠️  **Error** | {error} | {error/total*100:.1f}% |

**Status:** {'✅ ALL PASSED' if failed == 0 and error == 0 else '❌ SOME FAILED'}

---

## 📁 Results by Category

"""
    
    # Category breakdown
    for category in ['Unit Tests', 'Integration Tests', 'End-to-End Tests']:
        if category in by_category:
            stats = by_category[category]
            cat_passed = stats.get('passed', 0)
            cat_failed = stats.get('failed', 0)
            cat_skipped = stats.get('skipped', 0)
            cat_error = stats.get('error', 0)
            cat_total = stats['total']
            
            status_icon = '✅' if cat_failed == 0 and cat_error == 0 else '❌'
            
            report += f"""### {status_icon} {category}

- **Total:** {cat_total}
- **Passed:** {cat_passed} ({cat_passed/cat_total*100:.1f}%)
- **Failed:** {cat_failed}
- **Skipped:** {cat_skipped}
- **Error:** {cat_error}

"""
    
    # File breakdown
    report += """---

## 📄 Results by File

"""
    
    for file_name in sorted(by_file.keys()):
        stats = by_file[file_name]
        file_passed = stats.get('passed', 0)
        file_failed = stats.get('failed', 0)
        file_skipped = stats.get('skipped', 0)
        file_error = stats.get('error', 0)
        file_total = stats['total']
        
        status_icon = '✅' if file_failed == 0 and file_error == 0 else '❌'
        
        report += f"""### {status_icon} `{file_name}`

- **Total:** {file_total} | **Passed:** {file_passed} | **Failed:** {file_failed} | **Skipped:** {file_skipped} | **Error:** {file_error}

"""
    
    # Failed tests details
    failed_tests = [t for t in tests if t.get('outcome') == 'failed']
    if failed_tests:
        report += """---

## ❌ Failed Tests Details

"""
        for test in failed_tests:
            nodeid = test.get('nodeid', 'unknown')
            call = test.get('call', {})
            longrepr = call.get('longrepr', 'No error message')
            
            report += f"""### `{nodeid}`

```
{longrepr[:500]}...
```

"""
    
    # Skipped tests
    skipped_tests = [t for t in tests if t.get('outcome') == 'skipped']
    if skipped_tests:
        report += f"""---

## ⏭️  Skipped Tests ({len(skipped_tests)})

"""
        for test in skipped_tests[:10]:  # Show first 10
            nodeid = test.get('nodeid', 'unknown')
            report += f"- `{nodeid}`\n"
        
        if len(skipped_tests) > 10:
            report += f"\n*... and {len(skipped_tests) - 10} more skipped tests*\n"
    
    # Critical tests status
    report += """---

## 🔥 Critical Tests Status

"""
    
    critical_tests = [
        ('test_random_labels_sanity', 'Random Labels Test'),
        ('test_lucky_split_detection', 'Lucky Split Detection'),
        ('test_eq_csi_noise_sweep', 'CSI Noise Sweep'),
        ('test_generalization_new_dataset', 'New Dataset Generalization'),
    ]
    
    for test_pattern, test_name in critical_tests:
        found = False
        status = '❓ Not Found'
        
        for test in tests:
            if test_pattern in test.get('nodeid', ''):
                found = True
                outcome = test.get('outcome', 'unknown')
                if outcome == 'passed':
                    status = '✅ PASSED'
                elif outcome == 'failed':
                    status = '❌ FAILED'
                elif outcome == 'skipped':
                    status = '⏭️  SKIPPED'
                break
        
        if not found:
            status = '❓ NOT FOUND'
        
        report += f"- **{test_name}:** {status}\n"
    
    report += f"""

---

## 📝 Notes

- Tests marked with `@pytest.mark.slow` may take several minutes
- Some tests require dataset files in `dataset/` directory
- Some tests require result files in `result/` directory
- Skipped tests are expected if prerequisites are missing

---

**Report generated by:** `run_all_tests_with_report.py`  
**JSON Report:** `test_results.json`
"""
    
    # Write report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Markdown report generated: {output_file}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("🧪 Test Execution and Report Generation")
    print("="*70 + "\n")
    
    # Run tests
    exit_code, output = run_tests()
    
    # Parse JSON report
    print("\n" + "="*70)
    print("📊 Parsing Results...")
    print("="*70)
    
    json_data = parse_json_report()
    
    if json_data:
        summary = json_data.get('summary', {})
        print(f"\n✅ Total: {summary.get('total', 0)}")
        print(f"✅ Passed: {summary.get('passed', 0)}")
        print(f"❌ Failed: {summary.get('failed', 0)}")
        print(f"⏭️  Skipped: {summary.get('skipped', 0)}")
        print(f"⚠️  Error: {summary.get('error', 0)}")
        print(f"⏱️  Duration: {summary.get('duration', 0):.2f}s")
    else:
        print("❌ No JSON report found")
    
    # Generate Markdown report
    print("\n" + "="*70)
    print("📝 Generating Markdown Report...")
    print("="*70)
    
    generate_markdown_report(json_data)
    
    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)
    print("\n📄 Reports generated:")
    print("   - test_results.json (JSON)")
    print("   - TEST_EXECUTION_REPORT.md (Markdown)")
    print()
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

