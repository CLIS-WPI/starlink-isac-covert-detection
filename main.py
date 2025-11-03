# ======================================
# Ã°Å¸â€œâ€ž main.py
# Purpose: Main training pipeline (uses pre-generated dataset)
# OPTIMIZED: Loads cached dataset instead of regenerating
# NEW: Added AUC flip check for diagnostics
# ======================================

# ======================================
# 📄 main.py
# Purpose: Main training pipeline (uses pre-generated dataset)
# OPTIMIZED: Loads cached dataset instead of regenerating
# NEW: Added AUC flip check for diagnostics
# MULTI-GPU: Uses both H100s for training (1.7x speedup)
# ======================================

import os
# 🔧 MULTI-GPU: Use both H100s for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Enable both GPUs

import pickle
import numpy as np
import model
import tensorflow as tf

# 🚀 Initialize Multi-GPU Strategy
print("="*60)
print("🚀 INITIALIZING MULTI-GPU STRATEGY")
print("="*60)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        print(f"✓ Multi-GPU enabled: {strategy.num_replicas_in_sync} GPUs (H100 × 2)")
        print(f"  → Expected speedup: 1.7-1.9x for model training")
    else:
        strategy = tf.distribute.get_strategy()  # Single GPU fallback
        print(f"⚠️ Only {len(gpus)} GPU detected, using single-device strategy")
except Exception as e:
    print(f"⚠️ GPU initialization failed: {e}")
    strategy = tf.distribute.get_strategy()  # CPU fallback
print("="*60 + "\n")

# Configuration
from config.settings import (
    init_directories,
    NUM_SAMPLES_PER_CLASS,
    NUM_SATELLITES_FOR_TDOA,
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR
)

# Utilities
from utils.logger import init_logger
from utils.gpu import init_gpu
from utils.plots import generate_all_plots

# Core modules
from core.isac_system import ISACSystem, NTN_MODELS_AVAILABLE
from core.dataset_generator import generate_dataset_multi_satellite
from core.feature_extraction import extract_features_and_split

# 🔧 FIX: Use enhanced localization pipeline with STNN+CAF+GDOP+IRLS
from core.localization_enhanced import run_enhanced_tdoa_localization
from core.localization import compute_crlb
# Legacy import kept for backward compatibility
# from core.localization import run_tdoa_localization

# Model
from model.detector import train_detector, evaluate_detector

# Metrics
from sklearn.metrics import roc_auc_score # Import for AUC check


def main():
    """Main execution pipeline (training only)."""
    
    # ===== Phase 1: Initialization =====
    print("="*60)
    print("ISAC COVERT DETECTION & LOCALIZATION PIPELINE")
    print("="*60)
    
    init_logger('output.txt')
    init_directories()
    init_gpu(gpu_index=0)  # Use GPU 0 for training
    
    # ===== Phase 2: Load Dataset =====
    dataset_path = (
        f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
        f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    )
    
    if os.path.exists(dataset_path):
        print(f"\n[Phase 1] Loading dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print("Ã¢Å“â€œ Dataset loaded from disk")
    else:
        print(f"\n[Phase 1] Dataset not found at {dataset_path}")
        print("Please run: python3 generate_dataset_parallel.py")
        print("Or run:     python3 run_optimized_pipeline.py")
        return
    
    # Initialize ISAC system (needed for localization)
    print("\n[Phase 1] Initializing ISAC System...")
    isac = ISACSystem()
    
    # ✅ Display STNN status
    if hasattr(isac, 'stnn_estimator') and isac.stnn_estimator is not None:
        print("\n[STNN] Status: ✓ ENABLED")
        print(f"  → Localization will use hybrid STNN-aid CAF method")
        print(f"  → Expected speedup: ~10x vs traditional GCC-PHAT")
    else:
        print("\n[STNN] Status: ✗ DISABLED (using traditional GCC-PHAT)")
        print(f"  → To enable: python3 main.py --train-stnn")
    
    # ===== Data Validation =====
    print("\n=== DATA VALIDATION ===")
    labels = dataset['labels']
    print(f"Benign samples: {np.sum(labels==0)}")
    print(f"Attack samples: {np.sum(labels==1)}")
    print(f"Attack samples with emitter locations: "
          f"{sum(1 for loc in dataset['emitter_locations'] if loc is not None)}")
    
    # Power diagnostics
    # --- FRIEND'S CHECK 1: Ensure power ratio is printed ---
    benign_idx = np.where(labels == 0)[0][:100]
    attack_idx = np.where(labels == 1)[0][:100]
    benign_power = np.mean([
        np.mean(np.abs(dataset['iq_samples'][i])**2) 
        for i in benign_idx
    ])
    attack_power = np.mean([
        np.mean(np.abs(dataset['iq_samples'][i])**2) 
        for i in attack_idx
    ])
    power_ratio = attack_power / benign_power
    print(f"Power ratio (attack/benign): {power_ratio:.4f}")
    
    if power_ratio < 1.1:
        print("Ã¢Å¡Â Ã¯Â¸Â WARNING: Power ratio too low - dataset may be biased!")
    # --- END CHECK 1 ---
    
    # ===== Phase 3: Feature Extraction & Split =====
    print("\n[Phase 2] Feature extraction and split...")
    Xs_tr, Xs_te, Xr_tr, Xr_te, Xstft_tr, Xstft_te, y_tr, y_te, idx_tr, idx_te = (
        extract_features_and_split(dataset)
    )
    
    # ===== Phase 4: Model Training =====
    print("\n[Phase 3] Training detector model...")
    model, hist, temperature = train_detector(
        Xs_tr, Xr_tr, y_tr, Xs_te, Xr_te, y_te, 
        strategy=strategy  # 🚀 Pass multi-GPU strategy
    )
    
    # ===== Phase 5: Model Evaluation =====
    print("\n[Phase 4] Evaluating detector...")
    y_prob, best_thr, f1_scores, t = evaluate_detector(model, Xs_te, Xr_te, y_te, temperature)
    
    # --- FRIEND'S CHECK 2: Check for flipped AUC ---
    print("\n=== AUC DIAGNOSTICS ===")
    auc_normal = roc_auc_score(y_te, y_prob)
    auc_flipped = roc_auc_score(y_te, 1.0 - y_prob)
    print(f"AUC (Normal): {auc_normal:.4f}")
    print(f"AUC (Flipped): {auc_flipped:.4f}")
    if auc_flipped > auc_normal + 0.1:
        print("Ã°Å¸â€™Â¡ HINT: AUC is likely flipped. Check model output or label definitions.")
    print("=======================")
    # --- END CHECK 2 ---

    # Binary predictions using best threshold
    y_hat = (y_prob > best_thr).astype(int)
    
    # ===== Phase 6: TDoA Localization =====
    print("\n[Phase 5] Running Enhanced TDoA/FDoA Localization...")
    print("  → Using: FAST mode (use all satellites, no selection or CAF)")
    
    # 🚀 FAST MODE: Use all satellites without selection (satellite selection causing bad geometry!)
    loc_errors, tp_sample_ids, tp_ests, tp_gts, info_list = run_enhanced_tdoa_localization(
        dataset, y_hat, y_te, idx_te, isac,
        use_satellite_selection=False,  # ⚡ DISABLED - Selection causing bad estimates!
        use_caf_refinement=False,        # ⚡ DISABLED - CAF is too slow (1+ hour/sample!)
        use_fdoa=False,                  # ⚡ Disabled (TDOA sufficient)
        verbose=False                    # ⚡ Disabled for cleaner output
    )
    
    # CRLB Analysis
    if loc_errors:
        crlb_results = compute_crlb(
            loc_errors, tp_sample_ids, tp_ests, tp_gts, dataset, isac
        )
        
        # Save CRLB results
        if crlb_results is not None:
            try:
                import pandas as pd
                df_crlb = pd.DataFrame({
                    'sample_id': crlb_results['sample_ids'],
                    'crlb_m': crlb_results['crlb_values'],
                    'achieved_error_m': crlb_results['achieved_errors'],
                    'GT_x': crlb_results['gt_x'],
                    'GT_y': crlb_results['gt_y'],
                    'EST_x': crlb_results['est_x'],
                    'EST_y': crlb_results['est_y']
                })
                csv_path = os.path.join(RESULT_DIR, 'crlb_values.csv')
                df_crlb.to_csv(csv_path, index=False)
                print(f"Ã¢Å“â€œ CRLB analysis saved to {csv_path}")
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Could not save CRLB CSV: {e}")
        
        # Save localization errors
        try:
            with open(f"{RESULT_DIR}/localization_errors.txt", "w") as f:
                for e in loc_errors:
                    f.write(f"{e:.4f}\n")
            print(f"Ã¢Å“â€œ Localization errors saved to {RESULT_DIR}/localization_errors.txt")
        except:
            pass
    
    # ===== Phase 7: Generate All Plots =====
    print("\n[Phase 6] Generating plots...")
    generate_all_plots(
        y_te, y_hat, y_prob, Xs_te, dataset, hist,
        best_thr, np.argmax(f1_scores), f1_scores, t,
        loc_errors, isac, save_dir=RESULT_DIR
    )
    
    # ===== Final Summary =====
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"Dataset: {NUM_SAMPLES_PER_CLASS*2} samples "
          f"({NUM_SAMPLES_PER_CLASS} per class)")
    print(f"Model: Dual-Input CNN (saved to {MODEL_DIR}/)")
    print(f"Results: All plots saved to {RESULT_DIR}/")
    print(f"Best Threshold: {best_thr:.4f} (F1={f1_scores[np.argmax(f1_scores)]:.4f})")
    
    if loc_errors:
        med = float(np.median(loc_errors))
        p90 = float(np.percentile(loc_errors, 90))
        print(f"Localization: Median Error = {med:.2f} m, 90th = {p90:.2f} m")
    
    print("="*60)
    print("Ã¢Å“â€œ Pipeline execution completed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ISAC Covert Detection & Localization Pipeline')
    parser.add_argument('--train-stnn', action='store_true', 
                        help='Train STNN models before main pipeline')
    parser.add_argument('--stnn-epochs', type=int, default=50,
                        help='Number of epochs for STNN training')
    
    args = parser.parse_args()
    
    # ✅ Optional: Train STNN first
    if args.train_stnn:
        print("\n" + "="*60)
        print("STNN TRAINING MODE")
        print("="*60)
        
        from core.train_stnn_localization import main_training_pipeline
        
        dataset_path = (
            f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
            f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
        )
        
        main_training_pipeline(
            dataset_path=dataset_path,
            epochs_tdoa=args.stnn_epochs,
            epochs_fdoa=args.stnn_epochs,
            batch_size=32,
            use_multi_gpu=True
        )
        
        print("\n✓ STNN training complete! Running main pipeline...\n")
    
    main()