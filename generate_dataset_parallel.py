# ======================================
# 📄 generate_dataset_parallel.py
# Purpose: Parallel dataset generation using 2 GPUs
# OPTIMIZED: Each GPU generates half of the dataset independently
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
    
    print(f"\n[GPU {gpu_id}] ========================================")
    print(f"[GPU {gpu_id}] Worker started")
    print(f"[GPU {gpu_id}] Samples: {start_idx} to {end_idx}")
    print(f"[GPU {gpu_id}] ========================================\n")
    
    start_time = time.time()
    
    # Initialize ISAC system
    print(f"[GPU {gpu_id}] Initializing ISAC system...")
    isac = ISACSystem()
    
    # Pre-generate topologies (if NTN)
    if config['use_ntn']:
        isac.precompute_topologies(count=config['topology_cache_size'])
    
    # Generate dataset subset
    num_samples = end_idx - start_idx
    print(f"\n[GPU {gpu_id}] Generating {num_samples} samples...")
    
    dataset = generate_dataset_multi_satellite(
        isac,
        num_samples_per_class=num_samples // 2,  # Half benign, half attack
        num_satellites=config['num_satellites'],
        ebno_db_range=config['ebno_range'],
        covert_rate_mbps_range=config['covert_rate_range']
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
        Merged dataset dictionary
    """
    print("\n[Main] Merging datasets...")
    
    merged = {
        'iq_samples': [],
        'csi': [],
        'radar_echo': [],
        'labels': [],
        'emitter_locations': [],
        'satellite_receptions': [],
        'tx_time_padded': []
    }
    
    for ds in datasets:
        gpu_id = ds.pop('_gpu_id')
        start_idx = ds.pop('_start_idx')
        end_idx = ds.pop('_end_idx')
        
        print(f"  Merging GPU {gpu_id} data (samples {start_idx}-{end_idx})...")
        
        # Merge arrays
        for key in ['iq_samples', 'csi', 'radar_echo', 'labels', 'tx_time_padded']:
            if key in ds:
                merged[key].append(ds[key])
        
        # Merge lists
        merged['emitter_locations'].extend(ds['emitter_locations'])
        merged['satellite_receptions'].extend(ds['satellite_receptions'])
    
    # Concatenate arrays
    print("  Concatenating arrays...")
    for key in ['iq_samples', 'csi', 'radar_echo', 'labels', 'tx_time_padded']:
        merged[key] = np.concatenate(merged[key], axis=0)
    
    # Copy metadata from first dataset
    merged['sampling_rate'] = datasets[0]['sampling_rate']
    
    print(f"✓ Merged dataset: {len(merged['labels'])} total samples")
    
    return merged


def main():
    """Main parallel dataset generation."""
    from config.settings import (
        NUM_SAMPLES_PER_CLASS,
        NUM_SATELLITES_FOR_TDOA,
        DATASET_DIR,
        USE_NTN_IF_AVAILABLE
    )
    
    total_samples = NUM_SAMPLES_PER_CLASS * 2  # benign + attack
    
    # Configuration
    config = {
        'num_satellites': NUM_SATELLITES_FOR_TDOA,
        'use_ntn': USE_NTN_IF_AVAILABLE,
        'topology_cache_size': 1000,  # Cache 1000 topologies per GPU
        'ebno_range': (5, 15),
        'covert_rate_range': (1, 50)
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