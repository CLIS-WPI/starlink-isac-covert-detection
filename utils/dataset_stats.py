"""
Dataset Statistics Utility
===========================
Comprehensive dataset analysis and visualization
"""

import numpy as np
import pickle


def print_dataset_statistics(dataset_dict, detailed=True):
    """
    Print comprehensive statistics about the generated dataset.
    
    Args:
        dataset_dict: Dictionary containing dataset arrays
        detailed: If True, print detailed statistics per class
    """
    print(f"\n{'='*70}")
    print("📊 DATASET STATISTICS REPORT")
    print(f"{'='*70}\n")
    
    # ===== Basic Info =====
    labels = dataset_dict.get('labels', [])
    n_total = len(labels)
    n_benign = np.sum(labels == 0)
    n_attack = np.sum(labels == 1)
    
    print("🔢 Sample Counts:")
    print(f"  Total samples:   {n_total}")
    print(f"  Benign (label=0): {n_benign} ({n_benign/n_total*100:.1f}%)")
    print(f"  Attack (label=1): {n_attack} ({n_attack/n_total*100:.1f}%)")
    print(f"  Balance ratio:    {min(n_benign, n_attack) / max(n_benign, n_attack):.3f}")
    
    # ===== Feature Dimensions =====
    print(f"\n📐 Feature Dimensions:")
    for key in dataset_dict.keys():
        if key == 'labels':
            continue
        arr = dataset_dict[key]
        if isinstance(arr, (list, np.ndarray)):
            if isinstance(arr, list) and len(arr) > 0:
                if hasattr(arr[0], 'shape'):
                    shape_info = f"List of {len(arr)} items, first shape: {arr[0].shape}"
                else:
                    shape_info = f"List of {len(arr)} items"
            else:
                shape_info = f"{np.array(arr).shape}" if len(arr) > 0 else "Empty"
            print(f"  {key:20s}: {shape_info}")
    
    # ===== Power Statistics =====
    if 'tx_grids' in dataset_dict:
        print(f"\n⚡ Power Statistics (tx_grids):")
        tx_grids = dataset_dict['tx_grids']
        
        benign_powers = []
        attack_powers = []
        
        for i, label in enumerate(labels):
            if i < len(tx_grids):
                grid = np.squeeze(tx_grids[i])
                power = np.mean(np.abs(grid)**2)
                if label == 0:
                    benign_powers.append(power)
                else:
                    attack_powers.append(power)
        
        benign_powers = np.array(benign_powers)
        attack_powers = np.array(attack_powers)
        
        print(f"\n  Benign samples (n={len(benign_powers)}):")
        print(f"    Mean power:   {benign_powers.mean():.6e}")
        print(f"    Std power:    {benign_powers.std():.6e}")
        print(f"    Min power:    {benign_powers.min():.6e}")
        print(f"    Max power:    {benign_powers.max():.6e}")
        print(f"    Median power: {np.median(benign_powers):.6e}")
        
        print(f"\n  Attack samples (n={len(attack_powers)}):")
        print(f"    Mean power:   {attack_powers.mean():.6e}")
        print(f"    Std power:    {attack_powers.std():.6e}")
        print(f"    Min power:    {attack_powers.min():.6e}")
        print(f"    Max power:    {attack_powers.max():.6e}")
        print(f"    Median power: {np.median(attack_powers):.6e}")
        
        # Power difference analysis
        mean_diff = attack_powers.mean() - benign_powers.mean()
        rel_diff = abs(mean_diff) / (benign_powers.mean() + 1e-12)
        
        print(f"\n  Power Difference Analysis:")
        print(f"    Absolute diff: {mean_diff:.6e}")
        print(f"    Relative diff: {rel_diff*100:.2f}%")
        print(f"    Status: ", end="")
        if 5.0 <= rel_diff*100 <= 15.0:
            print(f"✅ GOOD (detectable but subtle)")
        elif rel_diff*100 < 5.0:
            print(f"⚠️  LOW (may be hard to detect)")
        else:
            print(f"⚠️  HIGH (may be too obvious)")
        
        # Power histogram
        if detailed:
            print(f"\n  Power Distribution (Histogram):")
            print(f"    Range        Benign  Attack")
            print(f"    " + "-"*40)
            
            all_powers = np.concatenate([benign_powers, attack_powers])
            bins = np.linspace(all_powers.min(), all_powers.max(), 6)
            
            for i in range(len(bins)-1):
                benign_count = np.sum((benign_powers >= bins[i]) & (benign_powers < bins[i+1]))
                attack_count = np.sum((attack_powers >= bins[i]) & (attack_powers < bins[i+1]))
                print(f"    [{bins[i]:.2e}, {bins[i+1]:.2e})  {benign_count:4d}    {attack_count:4d}")
    
    # ===== Magnitude Statistics =====
    if 'tx_grids' in dataset_dict and detailed:
        print(f"\n📈 Magnitude Statistics (tx_grids):")
        
        benign_mags = []
        attack_mags = []
        
        for i, label in enumerate(labels):
            if i < len(tx_grids):
                grid = np.squeeze(tx_grids[i])
                mag = np.abs(grid)
                if label == 0:
                    benign_mags.append(mag.flatten())
                else:
                    attack_mags.append(mag.flatten())
        
        benign_mags = np.concatenate(benign_mags)
        attack_mags = np.concatenate(attack_mags)
        
        print(f"  Benign: mean={benign_mags.mean():.4f}, std={benign_mags.std():.4f}, "
              f"max={benign_mags.max():.4f}")
        print(f"  Attack: mean={attack_mags.mean():.4f}, std={attack_mags.std():.4f}, "
              f"max={attack_mags.max():.4f}")
    
    # ===== CSI Statistics =====
    if 'csi' in dataset_dict and detailed:
        print(f"\n📡 CSI Statistics:")
        csi_data = dataset_dict['csi']
        if len(csi_data) > 0:
            csi_sample = csi_data[0]
            print(f"  Shape per sample: {csi_sample.shape if hasattr(csi_sample, 'shape') else 'N/A'}")
            
            # Power in CSI
            benign_csi_power = []
            attack_csi_power = []
            
            for i, label in enumerate(labels):
                if i < len(csi_data):
                    csi = np.array(csi_data[i])
                    power = np.mean(np.abs(csi)**2)
                    if label == 0:
                        benign_csi_power.append(power)
                    else:
                        attack_csi_power.append(power)
            
            if benign_csi_power and attack_csi_power:
                print(f"  Benign CSI power: {np.mean(benign_csi_power):.6e} ± {np.std(benign_csi_power):.6e}")
                print(f"  Attack CSI power: {np.mean(attack_csi_power):.6e} ± {np.std(attack_csi_power):.6e}")
    
    # ===== Emitter Locations =====
    if 'emitter_locations' in dataset_dict:
        print(f"\n📍 Emitter Locations:")
        emitters = dataset_dict['emitter_locations']
        
        # 🔧 FIX: Check bounds to prevent index out of range
        attack_emitters = []
        for i, label in enumerate(labels):
            if label == 1 and i < len(emitters) and emitters[i] is not None:
                attack_emitters.append(emitters[i])
        
        if attack_emitters:
            attack_emitters = np.array(attack_emitters)
            print(f"  Attack emitters: {len(attack_emitters)} locations")
            print(f"    X range: [{attack_emitters[:, 0].min():.1f}, {attack_emitters[:, 0].max():.1f}] m")
            print(f"    Y range: [{attack_emitters[:, 1].min():.1f}, {attack_emitters[:, 1].max():.1f}] m")
            print(f"    Z range: [{attack_emitters[:, 2].min():.1f}, {attack_emitters[:, 2].max():.1f}] m")
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    print("✅ Dataset Statistics Summary:")
    print(f"  Total samples: {n_total}")
    print(f"  Class balance: {n_benign}/{n_attack} (benign/attack)")
    
    if 'tx_grids' in dataset_dict:
        print(f"  Power difference: {rel_diff*100:.2f}%")
    
    print(f"{'='*70}\n")


def analyze_dataset_file(filepath, detailed=True):
    """
    Load and analyze a dataset pickle file.
    
    Args:
        filepath: Path to the pickle file
        detailed: If True, print detailed statistics
    """
    print(f"Loading dataset from: {filepath}")
    
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Loaded successfully")
    
    print_dataset_statistics(dataset, detailed=detailed)


if __name__ == "__main__":
    import sys
    
    # Default dataset path
    dataset_path = "dataset/dataset_samples200_sats12.pkl"
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    try:
        analyze_dataset_file(dataset_path, detailed=True)
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        print(f"   Please run: python3 generate_dataset_parallel.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
