#!/usr/bin/env python3
"""
🔴 RED-LINE TEST (Item 8 & 10)
Test with very high amplitude to ensure pipeline correctness
"""

import sys
import os

print("="*70)
print("🔴 RED-LINE SANITY TEST")
print("="*70)
print()
print("This test generates dataset with the following settings:")
print("  - NUM_SAMPLES_PER_CLASS = 25 (total 50)")
print("  - COVERT_AMP = 0.80 (very high)")
print("  - ADD_NOISE = False")
print()
print("Expected: AUC > 0.95 (if failed → problem in feature/mask/axes)")
print("="*70)
print()

# Temporarily modify settings
import config.settings as settings

original_samples = settings.NUM_SAMPLES_PER_CLASS
original_amp = settings.COVERT_AMP
original_noise = settings.ADD_NOISE

print("📝 Backing up original settings...")
print(f"  NUM_SAMPLES_PER_CLASS: {original_samples}")
print(f"  COVERT_AMP: {original_amp}")
print(f"  ADD_NOISE: {original_noise}")
print()

# Apply test settings
settings.NUM_SAMPLES_PER_CLASS = 25
settings.COVERT_AMP = 0.80
settings.ADD_NOISE = False

print("🔧 Applying RED-LINE test settings...")
print(f"  NUM_SAMPLES_PER_CLASS: {settings.NUM_SAMPLES_PER_CLASS}")
print(f"  COVERT_AMP: {settings.COVERT_AMP}")
print(f"  ADD_NOISE: {settings.ADD_NOISE}")
print()

response = input("⚠️  This will regenerate the dataset. Continue? (yes/no): ")
if response.lower() != 'yes':
    print("Cancelled.")
    sys.exit(0)

print()
print("="*70)
print("Step 1: Generate dataset with RED-LINE settings")
print("="*70)
os.system("python3 generate_dataset_parallel.py")

print()
print("="*70)
print("Step 2: Spectral analysis")
print("="*70)
os.system("python3 debug_spectral_diff.py")

print()
print("="*70)
print("Step 3: Run detection")
print("="*70)
os.system("python3 main_detection.py")

print()
print("="*70)
print("🔴 RED-LINE TEST COMPLETE")
print("="*70)
print()
print("📊 Result evaluation:")
print("  ✅ AUC > 0.95 → Pipeline is correct")
print("  ❌ AUC < 0.95 → Problem in:")
print("      - Feature extraction (wrong axes?)")
print("      - Focus mask (incorrect alignment?)")
print("      - Data loading (old dataset?)")
print()
print("💡 To restore normal settings:")
print("   1. In config/settings.py:")
print(f"      NUM_SAMPLES_PER_CLASS = {original_samples}")
print(f"      COVERT_AMP = {original_amp}")
print(f"      ADD_NOISE = {original_noise}")
print("   2. Regenerate dataset")
print("="*70)
