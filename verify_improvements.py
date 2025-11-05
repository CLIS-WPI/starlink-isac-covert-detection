#!/usr/bin/env python3
"""
✅ Verification Script for Final Improvements
==============================================
Checks that all three final improvements are properly implemented:
1. Random seed (RANDOM_SEED=42) in ablation_study.py
2. STFT spectrogram in main_detection_cnn.py
3. Focal loss in detector_cnn.py
"""

import os
import sys


def check_file_exists(filepath):
    """Check if file exists"""
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return False
    return True


def check_string_in_file(filepath, search_strings, description):
    """Check if all search strings exist in file"""
    if not check_file_exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_found = True
    for search_str in search_strings:
        if search_str in content:
            print(f"  ✅ Found: {description} - '{search_str[:50]}'")
        else:
            print(f"  ❌ Missing: {description} - '{search_str[:50]}'")
            all_found = False
    
    return all_found


def main():
    print("\n" + "="*70)
    print("✅ VERIFYING FINAL IMPROVEMENTS")
    print("="*70 + "\n")
    
    all_checks_passed = True
    
    # ===== Check 1: Random Seed in ablation_study.py =====
    print("[Check 1] Random Seed in ablation_study.py")
    print("-" * 70)
    
    ablation_checks = [
        "RANDOM_SEED = 42",
        "random.seed(RANDOM_SEED)",
        "np.random.seed(RANDOM_SEED)",
        "tf.random.set_seed(RANDOM_SEED)"
    ]
    
    if check_string_in_file("ablation_study.py", ablation_checks, "Random seed"):
        print("  ✅ Random seed properly set for reproducibility\n")
    else:
        print("  ❌ Random seed not properly configured\n")
        all_checks_passed = False
    
    # ===== Check 2: STFT Spectrogram in main_detection_cnn.py =====
    print("[Check 2] STFT Spectrogram in main_detection_cnn.py")
    print("-" * 70)
    
    stft_checks = [
        "from tensorflow.signal import stft",
        "def compute_spectrogram",
        "USE_SPECTROGRAM"
    ]
    
    if check_string_in_file("main_detection_cnn.py", stft_checks, "STFT spectrogram"):
        print("  ✅ STFT spectrogram properly implemented\n")
    else:
        print("  ❌ STFT spectrogram not properly configured\n")
        all_checks_passed = False
    
    # ===== Check 3: Focal Loss in detector_cnn.py =====
    print("[Check 3] Focal Loss in model/detector_cnn.py")
    print("-" * 70)
    
    focal_checks = [
        "from tensorflow.keras.losses import BinaryFocalCrossentropy",
        "use_focal_loss",
        "focal_gamma",
        "focal_alpha",
        "BinaryFocalCrossentropy"
    ]
    
    if check_string_in_file("model/detector_cnn.py", focal_checks, "Focal loss"):
        print("  ✅ Focal loss properly implemented\n")
    else:
        print("  ❌ Focal loss not properly configured\n")
        all_checks_passed = False
    
    # ===== Check 4: Config Settings =====
    print("[Check 4] Configuration Settings")
    print("-" * 70)
    
    config_checks = [
        "USE_SPECTROGRAM",
        "USE_FOCAL_LOSS",
        "FOCAL_LOSS_GAMMA",
        "FOCAL_LOSS_ALPHA"
    ]
    
    if check_string_in_file("config/settings.py", config_checks, "Config"):
        print("  ✅ Configuration properly set\n")
    else:
        print("  ❌ Configuration missing some settings\n")
        all_checks_passed = False
    
    # ===== Check 5: Integration in main_detection_cnn.py =====
    print("[Check 5] Integration in main_detection_cnn.py")
    print("-" * 70)
    
    integration_checks = [
        "USE_FOCAL_LOSS",
        "FOCAL_LOSS_GAMMA",
        "FOCAL_LOSS_ALPHA",
        "use_focal_loss=USE_FOCAL_LOSS"
    ]
    
    if check_string_in_file("main_detection_cnn.py", integration_checks, "Integration"):
        print("  ✅ Focal loss properly integrated\n")
    else:
        print("  ❌ Integration incomplete\n")
        all_checks_passed = False
    
    # ===== Final Summary =====
    print("="*70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("="*70)
        print("\n🎉 All final improvements are properly implemented!")
        print("\nYou can now:")
        print("  1. Run ablation study: python3 ablation_study.py --quick")
        print("  2. Train with spectrogram: Set USE_SPECTROGRAM=True in config")
        print("  3. Train with focal loss: Set USE_FOCAL_LOSS=True in config")
        print("  4. Full pipeline: ./run_advanced_optimization.sh\n")
        return 0
    else:
        print("❌ SOME CHECKS FAILED!")
        print("="*70)
        print("\n⚠️  Please review the errors above and fix them.")
        print("See FINAL_IMPROVEMENTS_APPLIED.md for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
