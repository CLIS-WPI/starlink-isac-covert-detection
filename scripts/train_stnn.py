# ======================================
# 📄 scripts/train_stnn.py
# Purpose: Convenient script to train STNN models
# Usage: python3 scripts/train_stnn.py
# ======================================

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.train_stnn_localization import main_training_pipeline
from config.settings import DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         STNN TRAINING FOR TDOA/FDOA ESTIMATION             ║
    ║  Based on: "A High-efficiency TDOA and FDOA Estimation     ║
    ║             Method Based on CNNs" (ICCIP 2024)             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Dataset path
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    print(f"Dataset: {dataset_path}")
    print(f"Multi-GPU: Both H100 GPUs will be used for training\n")
    
    # Run training
    main_training_pipeline(
        dataset_path=dataset_path,
        epochs_tdoa=50,
        epochs_fdoa=50,
        batch_size=32,
        use_multi_gpu=False  # Disabled multi-GPU to avoid hanging
    )
    
    print("\n✓ Training complete! Next steps:")
    print("  1. Run: python3 main.py --use-stnn")
    print("  2. Compare performance with baseline")