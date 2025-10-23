# ======================================
# 📄 utils/gpu.py
# Purpose: Initialize GPU configuration, set memory growth, and force GPU1 usage.
# Ensures the code always runs on GPU1 if available and avoids OOM errors.
# ======================================

import tensorflow as tf

def init_gpu(gpu_index=1):
    """
    Initialize GPU configuration.
    - Select GPU by index (default: 1)
    - Enable memory growth
    - Print GPU information
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if gpu_index >= len(gpus):
            print(f"[GPU] Requested GPU index {gpu_index} not available. Using GPU 0 instead.")
            gpu_index = 0
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"[GPU] Using GPU:{gpu_index} -> {gpus[gpu_index]}")
        except RuntimeError as e:
            print(e)
    else:
        print('[GPU] No GPU available, using CPU fallback.')