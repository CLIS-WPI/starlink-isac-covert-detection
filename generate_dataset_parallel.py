# ======================================
# 📄 generate_dataset_parallel.py
# Purpose: Parallel dataset generation using 2 GPUs
# OPTIMIZED: Each GPU generates half of the dataset independently
# FEATURE: Persistent NTN topology cache for instant loading
# ======================================
#
# 🚀 Topology Cache Benefits:
#   - First run: ~2 min to generate 1000 topologies (saved to cache/ntn_topologies.pkl)
#   - Subsequent runs: <1 sec to load from cache
#   - Saves ~4 min total (2 min × 2 GPUs) per run!
#
# ======================================

import os
import sys

# Disable GPU for main process (workers will set their own)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from multiprocessing import Process, Queue
import pickle
import numpy as np
import time


def worker_gpu(gpu_id, start_idx, end_idx, config, queue):
    """
    Worker process for one GPU.
    
    Args:
        gpu_id: GPU device ID (0 or 1)
        start_idx: Starting sample index
        end_idx: Ending sample index
        config: Configuration dict
        queue: Multiprocessing queue for results
    """
    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Import after setting GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    from core.isac_system import ISACSystem
    from core.dataset_generator import generate_dataset_multi_satellite
    
    # Warm-up: Pre-load TLE cache for SGP4 module to avoid IO overhead
    TLE_CACHE = None
    if config.get('tle_path'):
        try:
            from core.leo_orbit import read_tle_file
            print(f"[GPU {gpu_id}] Warming up SGP4 module with TLE file: {config['tle_path']}")
            TLE_CACHE = read_tle_file(config['tle_path'])
            print(f"[GPU {gpu_id}] Loaded {len(TLE_CACHE)} TLE entries")
        except Exception as e:
            print(f"[GPU {gpu_id}] ⚠️ Warning: Failed to load TLE file → {e}")
    
    print(f"\n[GPU {gpu_id}] ========================================")
    print(f"[GPU {gpu_id}] Worker started")
    print(f"[GPU {gpu_id}] Samples: {start_idx} to {end_idx}")
    print(f"[GPU {gpu_id}] ========================================\n")
    
    start_time = time.time()
    
    # Initialize ISAC system
    print(f"[GPU {gpu_id}] Initializing ISAC system...")
    isac = ISACSystem()
    
    # Load or pre-generate topologies (if NTN)
    if config['use_ntn']:
        cache_path = config.get('topology_cache_path', 'cache/ntn_topologies.pkl')
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            print(f"[GPU {gpu_id}] Loading topology cache from {cache_path}...")
            if isac.load_topology_cache(cache_path):
                print(f"[GPU {gpu_id}] ✅ Topology cache loaded instantly!")
            else:
                # Load failed, regenerate
                print(f"[GPU {gpu_id}] Cache load failed, regenerating...")
                isac.precompute_topologies(count=config['topology_cache_size'])
                isac.save_topology_cache(cache_path)
        else:
            # First time: generate and save
            print(f"[GPU {gpu_id}] No cache found, generating topologies...")
            isac.precompute_topologies(count=config['topology_cache_size'])
            isac.save_topology_cache(cache_path)
    
    # Generate dataset subset
    num_samples = end_idx - start_idx
    print(f"\n[GPU {gpu_id}] Generating {num_samples} samples...")
    
    dataset = generate_dataset_multi_satellite(
        isac,
        num_samples_per_class=num_samples // 2,  # Half benign, half attack
        num_satellites=config['num_satellites'],
        ebno_db_range=config['ebno_range'],
        covert_rate_mbps_range=config['covert_rate_range'],
        tle_path=config.get('tle_path'),                          # Pass TLE path for real Starlink positions
        inject_attack_into_pathb=config.get('inject_attack_into_pathb', True)  # Attack in Path-B for attacked sat
    )
    
    # Add metadata
    dataset['_gpu_id'] = gpu_id
    dataset['_start_idx'] = start_idx
    dataset['_end_idx'] = end_idx
    
    elapsed = time.time() - start_time
    print(f"\n[GPU {gpu_id}] ========================================")
    print(f"[GPU {gpu_id}] Worker finished in {elapsed/60:.1f} minutes")
    print(f"[GPU {gpu_id}] Generated {len(dataset['labels'])} samples")
    print(f"[GPU {gpu_id}] ========================================\n")
    
    # Send result back
    queue.put((gpu_id, dataset))


def merge_datasets(datasets):
    """
    Merge multiple dataset dictionaries into one.
    
    Args:
        datasets: List of dataset dicts from workers
    
    Returns:
        dict: Fully merged dataset
    """
    print("\n[Main] Merging datasets...")

    # Initialize merged containers
    merged = {
        'iq_samples': [],
        'csi': [],
        'radar_echo': [],
        'labels': [],
        'emitter_locations': [],
        'satellite_receptions': [],
        'tx_time_padded': [],
        'rx_time_b_full': []
    }

    # === Merge from each worker ===
    for ds in datasets:
        gpu_id = ds.pop('_gpu_id')
        start_idx = ds.pop('_start_idx')
        end_idx = ds.pop('_end_idx')

        print(f"  Merging GPU {gpu_id} data (samples {start_idx}-{end_idx})...")

        # Merge all array-like fields (if they exist)
        for key in ['iq_samples', 'csi', 'radar_echo', 'labels', 'tx_time_padded', 'rx_time_b_full']:
            if key in ds and ds[key] is not None:
                merged[key].append(ds[key])

        # Merge list-like fields
        if 'emitter_locations' in ds:
            merged['emitter_locations'].extend(ds['emitter_locations'])
        if 'satellite_receptions' in ds:
            merged['satellite_receptions'].extend(ds['satellite_receptions'])

    # === Concatenate all numpy arrays safely ===
    print("  Concatenating arrays...")
    for key in ['iq_samples', 'csi', 'radar_echo', 'labels', 'tx_time_padded', 'rx_time_b_full']:
        if len(merged[key]) > 0:
            try:
                # Check if shapes are consistent
                shapes = [arr.shape for arr in merged[key]]
                if len(set([s[1:] for s in shapes])) > 1:
                    # Shapes differ - pad to maximum length
                    print(f"  ⚠️ Key '{key}' has varying shapes, padding to max length...")
                    max_length = max([s[1] for s in shapes])
                    padded = []
                    for arr in merged[key]:
                        if arr.shape[1] < max_length:
                            pad_width = [(0, 0)] + [(0, max_length - arr.shape[1])] + [(0, 0)] * (arr.ndim - 2)
                            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
                            padded.append(arr_padded)
                        else:
                            padded.append(arr)
                    merged[key] = np.concatenate(padded, axis=0)
                    print(f"  ✓ Padded '{key}' to shape {merged[key].shape}")
                else:
                    merged[key] = np.concatenate(merged[key], axis=0)
            except Exception as e:
                print(f"⚠️ Warning: Failed to concatenate key '{key}' → {e}")
                merged[key] = np.array(merged[key], dtype=object)
        else:
            merged[key] = np.array([], dtype=np.complex64)

    # === Metadata ===
    merged['sampling_rate'] = datasets[0].get('sampling_rate', None)
    total = len(merged['labels']) if 'labels' in merged else 0

    print(f"✓ Merged dataset: {total} total samples")
    print("✓ Included keys:", list(merged.keys()))

    return merged



def main():
    """Main parallel dataset generation."""
    from config.settings import (
        NUM_SAMPLES_PER_CLASS,
        NUM_SATELLITES_FOR_TDOA,
        DATASET_DIR,
        USE_NTN_IF_AVAILABLE,
        TLE_PATH
    )
    
    total_samples = NUM_SAMPLES_PER_CLASS * 2  # benign + attack
    
    # Configuration
    config = {
        'num_satellites': NUM_SATELLITES_FOR_TDOA,
        'use_ntn': USE_NTN_IF_AVAILABLE,
        'topology_cache_size': 1000,  # Cache 1000 topologies per GPU
        'topology_cache_path': 'cache/ntn_topologies.pkl',  # Persistent cache file
        'ebno_range': (15, 25),        # 🔧 INCREASED from (5,15) to (15,25) for better SNR
        'covert_rate_range': (1, 50),
        'tle_path': TLE_PATH,                      # TLE file path for real Starlink positions
        'inject_attack_into_pathb': True           # Inject covert attack into Path-B for attacked satellite
    }
    
    print("="*60)
    print("PARALLEL DATASET GENERATION")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Satellites: {config['num_satellites']}")
    print(f"Channel model: {'NTN (TR 38.811)' if config['use_ntn'] else 'Rayleigh'}")
    print(f"GPUs: 2 (parallel generation)")
    print("="*60)
    
    # Split work between 2 GPUs
    mid = total_samples // 2
    splits = [
        (0, 0, mid),           # GPU 0: samples 0 to mid
        (1, mid, total_samples) # GPU 1: samples mid to total
    ]
    
    # Start workers
    queue = Queue()
    processes = []
    
    start_time = time.time()
    
    for gpu_id, start, end in splits:
        p = Process(target=worker_gpu, args=(gpu_id, start, end, config, queue))
        p.start()
        processes.append(p)
    
    # Wait for completion and collect results
    results = []
    for _ in splits:
        results.append(queue.get())
    
    for p in processes:
        p.join()
    
    # Merge results (sort by GPU ID for consistency)
    datasets = [r[1] for r in sorted(results, key=lambda x: x[0])]
    merged_dataset = merge_datasets(datasets)
    
    # Save
    save_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    print(f"\n[Main] Saving dataset to {save_path}...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(merged_dataset, f)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PARALLEL GENERATION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(merged_dataset['labels'])}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Samples/min: {len(merged_dataset['labels']) / (total_time/60):.1f}")
    print(f"Saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()