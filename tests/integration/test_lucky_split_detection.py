"""
🔥 CRITICAL TEST: Lucky Split Detection
=======================================
This test checks if high AUC is due to a single "lucky" train/test split.

We test 20 different random splits and verify that AUC remains consistently high.
If only one split gives high AUC → overfitting to that specific split.

This test is GOLD for reviewers.
"""
import pytest
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.detector_cnn import CNNDetector
from config.settings import USE_FOCAL_LOSS, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA


def load_dataset_for_test(scenario='sat'):
    """Load dataset for testing."""
    import glob
    import re
    
    scenario_letter = 'a' if scenario == 'sat' else 'b'
    pattern = f'dataset/dataset_scenario_{scenario_letter}_*.pkl'
    candidates = glob.glob(pattern)
    
    if not candidates:
        raise FileNotFoundError(f"No dataset found for scenario {scenario}")
    
    def extract_samples(path):
        match = re.search(r'dataset_scenario_[ab]_(\d+)\.pkl', path)
        return int(match.group(1)) if match else 0
    
    candidates.sort(key=extract_samples, reverse=True)
    dataset_file = Path(candidates[0])
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract data
    if 'rx_grids' in dataset:
        X = np.array(dataset['rx_grids'])
        y = np.array(dataset['labels'])
    elif 'data' in dataset:
        X = np.array([item for item in dataset['data']])
        y = np.array([item['is_covert'] if isinstance(item, dict) else meta.get('is_covert', 0) 
                     for item in dataset.get('meta', [])])
    else:
        raise ValueError("Unknown dataset format")
    
    return X, y


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_splits_consistency():
    """
    🔥 CRITICAL: Test AUC across 20 different random splits.
    
    If AUC is consistently high across all splits → robust result.
    If only one split gives high AUC → lucky split (overfitting).
    """
    print("\n" + "="*70)
    print("🔥 LUCKY SPLIT DETECTION TEST")
    print("="*70)
    
    # Load dataset (Scenario B - should have high AUC)
    try:
        X, y = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found. Generate dataset first.")
    
    # Use subset for faster testing
    n_samples = min(2000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"  Using {len(X)} samples")
    print(f"  Testing 20 different random splits...")
    
    n_splits = 20
    aucs = []
    
    for split_idx in range(n_splits):
        # Different random seed for each split
        random_state = 42 + split_idx
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Further split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        # Train model
        detector = CNNDetector(
            use_csi=False,
            use_attention=True,
            random_state=42,  # Fixed model seed
            learning_rate=0.001,
            dropout_rate=0.3,
            use_focal_loss=USE_FOCAL_LOSS,
            focal_gamma=FOCAL_LOSS_GAMMA,
            focal_alpha=FOCAL_LOSS_ALPHA
        )
        
        detector.train(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            epochs=15,  # Fewer epochs for speed
            batch_size=128,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = detector.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        aucs.append(auc)
        
        if (split_idx + 1) % 5 == 0:
            print(f"    Split {split_idx+1}/{n_splits}: AUC = {auc:.4f}")
    
    # Statistics
    aucs = np.array(aucs)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    min_auc = np.min(aucs)
    max_auc = np.max(aucs)
    
    print(f"\n  📊 Results across {n_splits} splits:")
    print(f"     Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"     Min AUC:  {min_auc:.4f}")
    print(f"     Max AUC:  {max_auc:.4f}")
    print(f"     Range:    {max_auc - min_auc:.4f}")
    
    # CRITICAL ASSERTIONS
    
    # 1. Mean should be high (for Scenario B)
    assert mean_auc >= 0.90, \
        f"Mean AUC ({mean_auc:.4f}) should be ≥ 0.90 for Scenario B. " \
        f"Current result suggests inconsistency."
    
    # 2. Standard deviation should be low (consistent performance)
    assert std_auc <= 0.10, \
        f"Std AUC ({std_auc:.4f}) should be ≤ 0.10. " \
        f"High variance suggests lucky splits. Current std is too high."
    
    # 3. Min AUC should still be reasonable
    assert min_auc >= 0.80, \
        f"Min AUC ({min_auc:.4f}) should be ≥ 0.80. " \
        f"Even worst split should perform well."
    
    # 4. Range should be small
    assert (max_auc - min_auc) <= 0.15, \
        f"AUC range ({max_auc - min_auc:.4f}) should be ≤ 0.15. " \
        f"Large range suggests lucky splits."
    
    print(f"  ✅ PASS: AUC is consistently high across all splits")
    print(f"     No evidence of 'lucky split' artifact")


@pytest.mark.integration
@pytest.mark.slow
def test_single_split_vs_multiple_splits():
    """
    Compare single split (80/20) with multiple splits to detect overfitting.
    """
    print("\n" + "="*70)
    print("🔥 SINGLE SPLIT vs MULTIPLE SPLITS")
    print("="*70)
    
    try:
        X, y = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found.")
    
    n_samples = min(2000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    # Single split (like in paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    detector = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector.train(
        X_train_final, y_train_final,
        X_val=X_val, y_val=y_val,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred = detector.predict(X_test)
    auc_single = roc_auc_score(y_test, y_pred)
    
    # Multiple splits (10 splits)
    n_splits = 10
    aucs_multi = []
    
    for split_idx in range(n_splits):
        random_state = 100 + split_idx
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        detector = CNNDetector(use_csi=False, use_attention=True, random_state=42)
        detector.train(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            epochs=15, batch_size=128, verbose=0
        )
        y_pred = detector.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        aucs_multi.append(auc)
    
    mean_multi = np.mean(aucs_multi)
    std_multi = np.std(aucs_multi)
    
    print(f"\n  📊 Comparison:")
    print(f"     Single split:    AUC = {auc_single:.4f}")
    print(f"     Multiple splits: AUC = {mean_multi:.4f} ± {std_multi:.4f}")
    
    # Single split should be close to mean of multiple splits
    diff = abs(auc_single - mean_multi)
    assert diff <= 0.10, \
        f"Single split AUC ({auc_single:.4f}) differs too much from " \
        f"mean of multiple splits ({mean_multi:.4f}). Difference: {diff:.4f}"
    
    print(f"  ✅ PASS: Single split is consistent with multiple splits")

