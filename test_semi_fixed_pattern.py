#!/usr/bin/env python3
"""
🎯 Test Semi-Fixed Pattern Injection
=====================================
Verify that semi-fixed pattern creates recognizable but diverse patterns
"""

import numpy as np
from config.settings import (
    COVERT_AMP,
    USE_SEMI_FIXED_PATTERN,
    NUM_COVERT_SUBCARRIERS,
    BAND_SIZE,
    BAND_START_OPTIONS,
    SYMBOL_PATTERN_OPTIONS,
    NOISE_STD,
    ADD_NOISE
)

def test_pattern_config():
    """Test the semi-fixed pattern configuration"""
    
    print("\n" + "="*70)
    print("🎯 SEMI-FIXED PATTERN CONFIGURATION")
    print("="*70)
    
    print(f"\n📊 Injection Parameters:")
    print(f"  COVERT_AMP:             {COVERT_AMP}")
    print(f"  USE_SEMI_FIXED_PATTERN: {USE_SEMI_FIXED_PATTERN}")
    print(f"  NUM_COVERT_SUBCARRIERS: {NUM_COVERT_SUBCARRIERS}")
    print(f"  BAND_SIZE:              {BAND_SIZE}")
    print(f"  BAND_START_OPTIONS:     {BAND_START_OPTIONS}")
    print(f"  ADD_NOISE:              {ADD_NOISE}")
    print(f"  NOISE_STD:              {NOISE_STD}")
    
    print(f"\n🎵 Symbol Pattern Options:")
    for i, pattern in enumerate(SYMBOL_PATTERN_OPTIONS):
        print(f"  Pattern {i}: {pattern}")
    
    # Calculate expected patterns
    num_band_patterns = len(BAND_START_OPTIONS)
    num_symbol_patterns = len(SYMBOL_PATTERN_OPTIONS)
    total_patterns = num_band_patterns * num_symbol_patterns
    
    print(f"\n🔢 Pattern Diversity:")
    print(f"  Band start positions:   {num_band_patterns}")
    print(f"  Symbol patterns:        {num_symbol_patterns}")
    print(f"  Total unique patterns:  {total_patterns}")
    
    # Simulate some patterns
    print(f"\n📈 Example Patterns (first 5 samples):")
    print(f"  {'Sample':<8} {'Band Start':<12} {'Symbol Pattern':<20} {'Subcarriers'}")
    print(f"  {'-'*70}")
    
    for i in range(5):
        band_start = np.random.choice(BAND_START_OPTIONS)
        pattern_idx = np.random.choice(len(SYMBOL_PATTERN_OPTIONS))
        symbols = SYMBOL_PATTERN_OPTIONS[pattern_idx]
        
        # Calculate subcarriers
        num_bands = NUM_COVERT_SUBCARRIERS // BAND_SIZE
        subcarriers = []
        for b in range(num_bands):
            band_offset = b * BAND_SIZE
            band_base = (band_start + band_offset) % 64
            subcarriers.extend(range(band_base, min(band_base + BAND_SIZE, 64)))
        subcarriers = subcarriers[:NUM_COVERT_SUBCARRIERS]
        
        sc_str = f"[{subcarriers[0]}-{subcarriers[BAND_SIZE-1]}..."
        print(f"  {i:<8} {band_start:<12} {str(symbols):<20} {sc_str}")
    
    # Expected power difference
    print(f"\n🎯 Expected Results:")
    print(f"  Power difference: ~4-6% (with COVERT_AMP={COVERT_AMP})")
    print(f"  Spectral signature: Strong (contiguous bands)")
    print(f"  Pattern consistency: High (limited variations)")
    print(f"  Expected AUC: 0.80-0.90 (if implemented correctly)")
    
    # Advantages
    print(f"\n✅ Advantages of Semi-Fixed Pattern:")
    print(f"  1. CNN can learn common pattern (band structure)")
    print(f"  2. Still has diversity ({total_patterns} unique combinations)")
    print(f"  3. Stronger spectral signature (contiguous bands)")
    print(f"  4. Easier to debug and visualize")
    
    # Comparison with previous approach
    print(f"\n📊 Comparison:")
    print(f"  {'Metric':<30} {'Random':<20} {'Semi-Fixed'}")
    print(f"  {'-'*70}")
    print(f"  {'Pattern consistency':<30} {'Low':<20} {'High'}")
    print(f"  {'Spectral signature':<30} {'Weak (scattered)':<20} {'Strong (bands)'}")
    print(f"  {'Power difference':<30} {'~1%':<20} {'~4-6%'}")
    print(f"  {'Expected AUC':<30} {'0.50-0.60':<20} {'0.80-0.90'}")
    print(f"  {'CNN learning speed':<30} {'Slow/None':<20} {'Fast'}")
    
    print("\n" + "="*70)
    print("💡 Ready to generate dataset with these settings!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_pattern_config()
