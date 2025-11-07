#!/usr/bin/env python3
"""
🔒 Reproducibility Utilities
============================
Centralized seed management and determinism settings for reproducibility.

Phase 0: Infrastructure hardening for evaluation.
"""

import os
import random
import numpy as np

# Lazy import TensorFlow (may not be available in all environments)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from config.settings import GLOBAL_SEED, CUDA_DETERMINISTIC


def set_global_seeds(seed=None, deterministic=True):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed (default: GLOBAL_SEED from settings)
        deterministic: Enable deterministic operations (may reduce performance)
    """
    if seed is None:
        seed = GLOBAL_SEED
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow (if available)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        
        # cuDNN determinism (for GPU operations)
        if deterministic and CUDA_DETERMINISTIC:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            # Note: This may reduce performance but ensures reproducibility
            print(f"  ✓ Deterministic mode enabled (seed={seed})")
        else:
            # Allow non-deterministic operations for better performance
            os.environ.pop('TF_DETERMINISTIC_OPS', None)
            os.environ.pop('TF_CUDNN_DETERMINISTIC', None)
            print(f"  ✓ Non-deterministic mode (seed={seed}, faster but less reproducible)")
        
        print(f"  ✓ Global seeds set: Python={seed}, NumPy={seed}, TensorFlow={seed}")
    else:
        print(f"  ✓ Global seeds set: Python={seed}, NumPy={seed} (TensorFlow not available)")


def set_worker_seed(base_seed, worker_id, gpu_id=0):
    """
    Set deterministic seed for a worker process.
    
    Args:
        base_seed: Base seed (typically GLOBAL_SEED)
        worker_id: Worker ID within GPU
        gpu_id: GPU ID
    
    Returns:
        worker_seed: Computed seed for this worker
    """
    worker_seed = base_seed + gpu_id * 1000 + worker_id * 100
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if TF_AVAILABLE:
        tf.random.set_seed(worker_seed)
    return worker_seed


def log_seed_info(script_name, seed=None):
    """
    Log seed information at the start of a script.
    
    Args:
        script_name: Name of the script
        seed: Seed value (default: GLOBAL_SEED)
    """
    if seed is None:
        seed = GLOBAL_SEED
    
    print("="*70)
    print(f"🔒 REPRODUCIBILITY INFO: {script_name}")
    print("="*70)
    print(f"  Global Seed: {seed}")
    print(f"  CUDA Deterministic: {CUDA_DETERMINISTIC}")
    print(f"  TF_DETERMINISTIC_OPS: {os.environ.get('TF_DETERMINISTIC_OPS', 'Not set')}")
    print(f"  TF_CUDNN_DETERMINISTIC: {os.environ.get('TF_CUDNN_DETERMINISTIC', 'Not set')}")
    print("="*70)

