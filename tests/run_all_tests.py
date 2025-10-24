"""
Run all test cases
"""
import pytest
import sys
import os

def main():
    """Run all tests with optional coverage."""
    
    # Check if pytest-cov is installed
    try:
        import pytest_cov
        has_coverage = True
    except ImportError:
        has_coverage = False
        print("⚠️ pytest-cov not installed. Running tests without coverage.")
        print("   Install with: pip install pytest-cov --break-system-packages\n")
    
    # Base arguments
    args = [
        'tests/',
        '-v',
        '--tb=short',
        '--color=yes',
        '-x'  # Stop on first failure
    ]
    
    # Add coverage if available
    if has_coverage:
        args.extend([
            '--cov=core',
            '--cov=model',
            '--cov=utils',
            '--cov-report=term-missing',
            '--cov-report=html'
        ])
    
    print("="*60)
    print("RUNNING TEST SUITE")
    print("="*60)
    
    # Run tests
    exit_code = pytest.main(args)
    
    print("\n" + "="*60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
        if has_coverage:
            print("📊 Coverage report saved to: htmlcov/index.html")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())