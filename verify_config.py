#!/usr/bin/env python3
"""
✅ Verify Semi-Fixed Pattern Configuration
===========================================
Check that settings are correctly configured for semi-fixed pattern
"""

from config.settings import (
    USE_SEMI_FIXED_PATTERN,
    RANDOMIZE_SUBCARRIERS,
    RANDOMIZE_SYMBOLS,
    COVERT_AMP,
    BAND_SIZE,
    BAND_START_OPTIONS,
    SYMBOL_PATTERN_OPTIONS,
    ADD_NOISE,
    NOISE_STD,
    NUM_COVERT_SUBCARRIERS
)

def verify_config():
    """Verify that configuration is correct for semi-fixed pattern"""
    
    print("\n" + "="*70)
    print("✅ SEMI-FIXED PATTERN CONFIGURATION VERIFICATION")
    print("="*70)
    
    all_good = True
    
    # Check 1: Semi-fixed enabled
    print(f"\n1️⃣ Semi-Fixed Pattern:")
    print(f"   USE_SEMI_FIXED_PATTERN = {USE_SEMI_FIXED_PATTERN}")
    if USE_SEMI_FIXED_PATTERN:
        print(f"   ✅ Enabled")
    else:
        print(f"   ❌ ERROR: Should be True!")
        all_good = False
    
    # Check 2: CRITICAL - Randomization must be disabled!
    print(f"\n2️⃣ Randomization (MUST be False):")
    print(f"   RANDOMIZE_SUBCARRIERS = {RANDOMIZE_SUBCARRIERS}")
    print(f"   RANDOMIZE_SYMBOLS = {RANDOMIZE_SYMBOLS}")
    
    if not RANDOMIZE_SUBCARRIERS and not RANDOMIZE_SYMBOLS:
        print(f"   ✅ Correctly disabled - semi-fixed will work!")
    elif RANDOMIZE_SUBCARRIERS or RANDOMIZE_SYMBOLS:
        print(f"   ❌ ERROR: These MUST be False for semi-fixed pattern!")
        print(f"   ⚠️  CNN cannot learn with random patterns!")
        all_good = False
    
    # Check 3: Pattern parameters
    print(f"\n3️⃣ Pattern Parameters:")
    print(f"   COVERT_AMP = {COVERT_AMP}")
    print(f"   NUM_COVERT_SUBCARRIERS = {NUM_COVERT_SUBCARRIERS}")
    print(f"   BAND_SIZE = {BAND_SIZE}")
    print(f"   BAND_START_OPTIONS = {BAND_START_OPTIONS}")
    
    expected_patterns = len(BAND_START_OPTIONS) * len(SYMBOL_PATTERN_OPTIONS)
    print(f"   Total unique patterns = {expected_patterns}")
    
    if 1.7 <= COVERT_AMP <= 2.0:
        print(f"   ✅ COVERT_AMP in optimal range (1.7-2.0)")
    else:
        print(f"   ⚠️  COVERT_AMP outside optimal range")
    
    if expected_patterns >= 4 and expected_patterns <= 16:
        print(f"   ✅ Good number of patterns (4-16)")
    else:
        print(f"   ⚠️  Pattern count may be suboptimal")
    
    # Check 4: Symbol patterns
    print(f"\n4️⃣ Symbol Patterns:")
    for i, pattern in enumerate(SYMBOL_PATTERN_OPTIONS):
        print(f"   Pattern {i}: {pattern}")
    print(f"   ✅ {len(SYMBOL_PATTERN_OPTIONS)} patterns defined")
    
    # Check 5: Noise
    print(f"\n5️⃣ Noise Configuration:")
    print(f"   ADD_NOISE = {ADD_NOISE}")
    print(f"   NOISE_STD = {NOISE_STD}")
    
    if ADD_NOISE and 0.01 <= NOISE_STD <= 0.02:
        print(f"   ✅ Optimal noise level for robustness")
    elif not ADD_NOISE:
        print(f"   ⚠️  Noise disabled - model may overfit")
    else:
        print(f"   ⚠️  Noise level may be suboptimal")
    
    # Expected results
    print(f"\n" + "="*70)
    print("🎯 EXPECTED RESULTS WITH THIS CONFIGURATION")
    print("="*70)
    
    if all_good:
        print(f"\n✅ Configuration is CORRECT!")
        print(f"\n📊 Expected Performance:")
        print(f"   Power difference:  3-4%")
        print(f"   AUC:               0.75-0.85")
        print(f"   Convergence:       ~20 epochs")
        print(f"   Pattern diversity: {expected_patterns} unique patterns")
        print(f"   Spectral signature: Strong (contiguous bands)")
        
        print(f"\n💡 What CNN will learn:")
        print(f"   - Contiguous 8-subcarrier bands")
        print(f"   - {len(BAND_START_OPTIONS)} possible band positions: {BAND_START_OPTIONS}")
        print(f"   - {len(SYMBOL_PATTERN_OPTIONS)} symbol patterns: odd vs even")
        print(f"   - Recognizable pattern with controlled diversity")
        
    else:
        print(f"\n❌ Configuration has ERRORS!")
        print(f"\n🔧 Required fixes:")
        if RANDOMIZE_SUBCARRIERS or RANDOMIZE_SYMBOLS:
            print(f"   1. Set RANDOMIZE_SUBCARRIERS = False")
            print(f"   2. Set RANDOMIZE_SYMBOLS = False")
        if not USE_SEMI_FIXED_PATTERN:
            print(f"   3. Set USE_SEMI_FIXED_PATTERN = True")
    
    # Summary
    print(f"\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    checks = [
        ("Semi-fixed enabled", USE_SEMI_FIXED_PATTERN),
        ("Randomization disabled", not RANDOMIZE_SUBCARRIERS and not RANDOMIZE_SYMBOLS),
        ("COVERT_AMP optimal", 1.7 <= COVERT_AMP <= 2.0),
        ("Pattern count good", 4 <= expected_patterns <= 16),
        ("Noise configured", ADD_NOISE and 0.01 <= NOISE_STD <= 0.02),
    ]
    
    passed = sum(1 for _, check in checks if check)
    total = len(checks)
    
    print(f"\n✅ Checks passed: {passed}/{total}")
    
    for name, check in checks:
        status = "✅" if check else "❌"
        print(f"   {status} {name}")
    
    if all_good:
        print(f"\n🚀 Ready to generate dataset and train CNN!")
        print(f"\n   Next steps:")
        print(f"   1. rm -f dataset/dataset_samples*.pkl  # Remove old dataset")
        print(f"   2. python3 generate_dataset_parallel.py  # Generate new")
        print(f"   3. python3 main_detection_cnn.py         # Train CNN")
    else:
        print(f"\n⚠️  Fix configuration errors before proceeding!")
    
    print("\n" + "="*70 + "\n")
    
    return all_good


if __name__ == "__main__":
    import sys
    success = verify_config()
    sys.exit(0 if success else 1)
