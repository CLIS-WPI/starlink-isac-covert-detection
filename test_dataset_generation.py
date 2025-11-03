#!/usr/bin/env python3
"""
Test if dataset was regenerated with new code
Check for signature of new emitter location code
"""

import pickle
import numpy as np
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"

print("="*70)
print("DATASET GENERATION CHECK")
print("="*70)

try:
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"\n✓ Dataset loaded: {dataset_path}")
    print(f"  Total samples: {len(dataset['labels'])}")
    
    # Check if emitter_locations exist and are on ground
    emitter_locations = dataset.get('emitter_locations', [])
    
    if not emitter_locations:
        print("\n❌ FAIL: No emitter_locations in dataset!")
        print("   → Dataset generated with OLD code")
        exit(1)
    
    # Check attack samples
    labels = dataset['labels']
    attack_count = 0
    ground_count = 0
    satellite_count = 0
    
    for i, label in enumerate(labels):
        if label == 1:  # Attack
            attack_count += 1
            emitter = emitter_locations[i]
            if emitter is not None:
                z = emitter[2]
                if abs(z) < 100e3:  # Within 100 km of ground
                    ground_count += 1
                else:
                    satellite_count += 1
                    if satellite_count <= 5:  # Print first 5
                        print(f"  ⚠️  Attack sample {i}: emitter z = {z/1e3:.1f} km")
    
    print(f"\n📊 RESULTS:")
    print(f"  Attack samples: {attack_count}")
    print(f"  Emitters on ground (|z|<100km): {ground_count}")
    print(f"  Emitters in space (|z|≥100km): {satellite_count}")
    
    # Check attack signal variance
    iq_samples = dataset['iq_samples']
    attack_powers = []
    for i, label in enumerate(labels):
        if label == 1:
            power = np.mean(np.abs(iq_samples[i])**2)
            attack_powers.append(power)
    
    if len(attack_powers) > 1:
        attack_std = np.std(attack_powers)
        attack_mean = np.mean(attack_powers)
        cv = attack_std / attack_mean if attack_mean > 0 else 0
        
        print(f"\n📈 ATTACK SIGNAL VARIANCE:")
        print(f"  Mean power: {attack_mean:.6e}")
        print(f"  Std power: {attack_std:.6e}")
        print(f"  CV (std/mean): {cv:.4f}")
        
        if cv < 0.1:
            print(f"\n❌ FAIL: Attack signals have NO variance (CV={cv:.4f})")
            print("   → All attack samples identical!")
            print("   → Power normalization was applied (old dataset)")
            exit(1)
        elif cv > 3:
            print(f"\n✅ PASS: Attack signals have natural variance (CV={cv:.4f})")
            print("   → Power normalization NOT applied ✓")
        else:
            print(f"\n⚠️  WARNING: Attack variance moderate (CV={cv:.4f})")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT:")
    print("="*70)
    
    if satellite_count == 0 and ground_count > 0 and cv > 3:
        print("✅ PASS: Dataset generated with NEW code!")
        print("   → Emitters on ground ✓")
        print("   → Natural power variance ✓")
        print("   → Ready for training!")
    else:
        print("❌ FAIL: Dataset has issues!")
        if satellite_count > 0:
            print(f"   → {satellite_count} emitters in space (should be 0)")
        if cv < 0.1:
            print(f"   → Attack variance too low (CV={cv:.4f})")
        print("\n💡 Solution: Regenerate dataset!")
        print("   rm dataset/*.pkl")
        print("   python3 generate_dataset_parallel.py")
        exit(1)
    
    print("="*70)

except FileNotFoundError:
    print(f"\n❌ Dataset not found: {dataset_path}")
    print("   Generate with: python3 generate_dataset_parallel.py")
    exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
