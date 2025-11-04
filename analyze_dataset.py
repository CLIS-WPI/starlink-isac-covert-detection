#!/usr/bin/env python3
"""
Analyze Dataset Tool
====================
Standalone tool to analyze any generated dataset pickle file.

Usage:
    python3 analyze_dataset.py                          # Default path
    python3 analyze_dataset.py dataset/my_dataset.pkl   # Custom path
    python3 analyze_dataset.py --brief                   # Brief summary only
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_stats import analyze_dataset_file


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('dataset_path', nargs='?', 
                        default='dataset/dataset_samples200_sats12.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--brief', action='store_true',
                        help='Show brief summary only (no detailed stats)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset file not found: {args.dataset_path}")
        print(f"\nAvailable datasets in dataset/ directory:")
        
        if os.path.exists('dataset'):
            files = [f for f in os.listdir('dataset') if f.endswith('.pkl')]
            if files:
                for f in sorted(files):
                    size_mb = os.path.getsize(f'dataset/{f}') / (1024*1024)
                    print(f"  - {f} ({size_mb:.1f} MB)")
            else:
                print("  (none)")
        else:
            print("  Dataset directory not found.")
        
        print(f"\nPlease run: python3 generate_dataset_parallel.py")
        sys.exit(1)
    
    # Analyze dataset
    try:
        analyze_dataset_file(args.dataset_path, detailed=not args.brief)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
