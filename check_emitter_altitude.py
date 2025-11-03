#!/usr/bin/env python3
"""
Verify emitter locations are on ground (z≈0), not at satellite altitude
"""

import pickle
import numpy as np
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

def check_emitter_altitudes(dataset_path):
    """Check that attack emitters are on ground, not in space!"""
    
    print("="*70)
    print("EMITTER ALTITUDE VERIFICATION")
    print("="*70)
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    labels = dataset['labels']
    emitter_locations = dataset.get('emitter_locations', [])
    
    if not emitter_locations:
        print("❌ No emitter_locations in dataset!")
        return
    
    print(f"\n📊 Dataset: {len(labels)} samples")
    
    # Check attack samples
    attack_indices = [i for i, label in enumerate(labels) if label == 1]
    
    print(f"\n🎯 Checking {len(attack_indices)} attack samples...")
    print("-"*70)
    
    ground_emitters = 0
    satellite_emitters = 0
    
    emitter_altitudes = []
    
    for i in attack_indices[:20]:  # Check first 20
        emitter = emitter_locations[i]
        if emitter is not None:
            altitude = emitter[2]
            emitter_altitudes.append(altitude)
            
            if i < 10:  # Print first 10
                print(f"Sample {i}: emitter = [{emitter[0]/1e3:.1f}, {emitter[1]/1e3:.1f}, {altitude/1e3:.1f}] km")
            
            if abs(altitude) < 100e3:  # Within 100 km of ground
                ground_emitters += 1
            else:
                satellite_emitters += 1
    
    print("-"*70)
    
    # Statistics
    if emitter_altitudes:
        mean_alt = np.mean(emitter_altitudes)
        max_alt = np.max(np.abs(emitter_altitudes))
        
        print(f"\n📈 STATISTICS:")
        print(f"  Mean altitude: {mean_alt/1e3:.1f} km")
        print(f"  Max |altitude|: {max_alt/1e3:.1f} km")
        print(f"  Ground emitters (|z|<100km): {ground_emitters}")
        print(f"  Satellite emitters (|z|≥100km): {satellite_emitters}")
    
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT:")
    print("="*70)
    
    if satellite_emitters == 0 and ground_emitters > 0:
        print("✅ PASS: All emitters on ground (z≈0)")
        print("   → Realistic covert channel scenario")
    elif satellite_emitters > 0:
        print(f"❌ FAIL: {satellite_emitters} emitters at satellite altitude!")
        print("   → Unrealistic! Emitters should be on ground")
        print("   → Need to regenerate dataset with fixed code")
    else:
        print("⚠️  WARNING: Could not verify emitter altitudes")
    
    print("="*70)

if __name__ == "__main__":
    dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    
    try:
        check_emitter_altitudes(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Generate dataset first with: python3 generate_dataset_parallel.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
