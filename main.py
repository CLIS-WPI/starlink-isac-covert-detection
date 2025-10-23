# ======================================
# 📄 main.py
# Purpose: Main training pipeline (uses pre-generated dataset)
# OPTIMIZED: Loads cached dataset instead of regenerating
# ======================================

import os
import pickle
import numpy as np
import tensorflow as tf

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
from core.localization import run_tdoa_localization, compute_crlb

# Model
from model.detector import train_detector, evaluate_detector


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
        print("✓ Dataset loaded from disk")
    else:
        print(f"\n[Phase 1] Dataset not found at {dataset_path}")
        print("Please run: python3 generate_dataset_parallel.py")
        print("Or run:     python3 run_optimized_pipeline.py")
        return
    
    # Initialize ISAC system (needed for localization)
    print("\n[Phase 1] Initializing ISAC System...")
    isac = ISACSystem()
    
    # ===== Data Validation =====
    print("\n=== DATA VALIDATION ===")
    labels = dataset['labels']
    print(f"Benign samples: {np.sum(labels==0)}")
    print(f"Attack samples: {np.sum(labels==1)}")
    print(f"Attack samples with emitter locations: "
          f"{sum(1 for loc in dataset['emitter_locations'] if loc is not None)}")
    
    # Power diagnostics
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
        print("⚠️ WARNING: Power ratio too low - dataset may be biased!")
    
    # ===== Phase 3: Feature Extraction & Split =====
    print("\n[Phase 2] Feature extraction and split...")
    Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te = (
        extract_features_and_split(dataset)
    )
    
    # ===== Phase 4: Model Training =====
    print("\n[Phase 3] Training detector model...")
    model, hist = train_detector(Xs_tr, Xr_tr, y_tr, Xs_te, Xr_te, y_te)
    
    # ===== Phase 5: Model Evaluation =====
    print("\n[Phase 4] Evaluating detector...")
    y_prob, best_thr, f1_scores, t = evaluate_detector(model, Xs_te, Xr_te, y_te)
    
    # Binary predictions using best threshold
    y_hat = (y_prob > best_thr).astype(int)
    
    # ===== Phase 6: TDoA Localization =====
    print("\n[Phase 5] Running TDoA localization...")
    loc_errors, tp_sample_ids, tp_ests, tp_gts = run_tdoa_localization(
        dataset, y_hat, y_te, idx_te, isac
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
                print(f"✓ CRLB analysis saved to {csv_path}")
            except Exception as e:
                print(f"⚠️ Could not save CRLB CSV: {e}")
        
        # Save localization errors
        try:
            with open(f"{RESULT_DIR}/localization_errors.txt", "w") as f:
                for e in loc_errors:
                    f.write(f"{e:.4f}\n")
            print(f"✓ Localization errors saved to {RESULT_DIR}/localization_errors.txt")
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
    print("✓ Pipeline execution completed successfully")


if __name__ == "__main__":
    main()