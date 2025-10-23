# ======================================
# 📄 run_optimized_pipeline.py
# Purpose: Optimized 2-stage pipeline (parallel gen + single training)
# ======================================

import os
import subprocess
import time
import sys


def check_dataset_exists():
    """Check if dataset already exists."""
    from config.settings import NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA, DATASET_DIR
    
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    return os.path.exists(dataset_path), dataset_path


def main():
    """Run optimized 2-stage pipeline."""
    
    print("="*60)
    print("OPTIMIZED ISAC PIPELINE (2-STAGE)")
    print("="*60)
    print("Stage 1: Parallel dataset generation (2 GPUs)")
    print("Stage 2: Model training (1 GPU)")
    print("="*60)
    
    overall_start = time.time()
    
    # ===== Stage 1: Dataset Generation =====
    exists, dataset_path = check_dataset_exists()
    
    if exists:
        print(f"\n✓ Dataset already exists: {dataset_path}")
        print("  Skipping generation. Delete file to regenerate.")
        gen_time = 0
    else:
        print("\n[Stage 1/2] Parallel Dataset Generation")
        print("-"*60)
        gen_start = time.time()
        
        try:
            result = subprocess.run(
                ['python3', 'generate_dataset_parallel.py'],
                check=True,
                capture_output=False  # Show output in real-time
            )
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Dataset generation failed with exit code {e.returncode}")
            sys.exit(1)
        
        gen_time = time.time() - gen_start
        print(f"\n✓ Stage 1 completed in {gen_time/60:.1f} minutes")
    
    # ===== Stage 2: Training =====
    print("\n[Stage 2/2] Model Training & Evaluation")
    print("-"*60)
    train_start = time.time()
    
    try:
        result = subprocess.run(
            ['python3', 'main.py'],
            check=True,
            capture_output=False  # Show output in real-time
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    
    train_time = time.time() - train_start
    print(f"\n✓ Stage 2 completed in {train_time/60:.1f} minutes")
    
    # ===== Summary =====
    total_time = time.time() - overall_start
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Stage 1 (Generation): {gen_time/60:.1f} minutes")
    print(f"Stage 2 (Training):   {train_time/60:.1f} minutes")
    print(f"Total time:           {total_time/60:.1f} minutes")
    print("="*60)
    print("\n✓ All results saved to:")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Model:   model/best_model.keras")
    print(f"  - Plots:   result/*.pdf")
    print("="*60)


if __name__ == "__main__":
    main()