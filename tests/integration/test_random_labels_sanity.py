"""
🔥 CRITICAL TEST: Random Labels Sanity Check
============================================
This test ensures the model is not learning "fake" patterns or suffering from data leakage.

If AUC remains high with random labels → strong indication of data leakage/bug.

This is one of the most important scientific tests for validating results.
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
def test_random_labels_should_fail():
    """
    🔥 CRITICAL: Test that model fails with random labels.
    
    If AUC > 0.6 with random labels → data leakage or bug!
    Expected: AUC ≈ 0.5 (random guessing)
    """
    print("\n" + "="*70)
    print("🔥 RANDOM LABELS SANITY TEST")
    print("="*70)
    
    # Load dataset (Scenario B for best case)
    try:
        X, y_real = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found. Generate dataset first.")
    
    # Use subset for faster testing
    n_samples = min(1000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y_real = y_real[indices]
    
    print(f"  Using {len(X)} samples for test")
    
    # Shuffle labels randomly (destroy signal)
    np.random.seed(42)
    y_random = np.random.permutation(y_real.copy())
    
    print(f"  Original labels: {np.sum(y_real==0)} benign, {np.sum(y_real==1)} attack")
    print(f"  Random labels: {np.sum(y_random==0)} benign, {np.sum(y_random==1)} attack")
    
    # Split
    X_train, X_test, y_train_rand, y_test_rand = train_test_split(
        X, y_random, test_size=0.2, random_state=42, stratify=y_random
    )
    
    # Further split for validation
    X_train_final, X_val, y_train_rand_final, y_val_rand = train_test_split(
        X_train, y_train_rand, test_size=0.2, random_state=42, stratify=y_train_rand
    )
    
    # Train model with RANDOM labels
    print("  🚀 Training CNN with RANDOM labels...")
    detector = CNNDetector(
        use_csi=False,
        use_attention=True,
        random_state=42,
        learning_rate=0.001,
        dropout_rate=0.3,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_LOSS_GAMMA,
        focal_alpha=FOCAL_LOSS_ALPHA
    )
    
    detector.train(
        X_train_final, y_train_rand_final,
        X_val=X_val,
        y_val=y_val_rand,
        epochs=20,  # Fewer epochs for sanity test
        batch_size=128,
        verbose=0
    )
    
    # Evaluate
    y_pred_proba = detector.predict(X_test)
    auc_random = roc_auc_score(y_test_rand, y_pred_proba)
    
    print(f"\n  📊 Results with RANDOM labels:")
    print(f"     AUC = {auc_random:.4f}")
    
    # CRITICAL ASSERTION: AUC should be close to 0.5 (random)
    # If AUC > 0.6, there's likely data leakage or a bug
    assert auc_random < 0.60, \
        f"❌ CRITICAL: AUC={auc_random:.4f} with random labels! " \
        f"This indicates data leakage or bug. Expected AUC ≈ 0.5"
    
    # Also check that it's not too low (should be around 0.5)
    assert auc_random > 0.40, \
        f"⚠️  AUC={auc_random:.4f} is suspiciously low. Expected ≈ 0.5"
    
    print(f"  ✅ PASS: AUC={auc_random:.4f} is close to random (0.5)")
    print(f"     This confirms no data leakage or bug")


@pytest.mark.integration
@pytest.mark.slow
def test_real_labels_vs_random_comparison():
    """
    Compare performance with real labels vs random labels.
    
    Real labels should achieve high AUC, random should be ~0.5.
    """
    print("\n" + "="*70)
    print("🔥 REAL vs RANDOM LABELS COMPARISON")
    print("="*70)
    
    try:
        X, y_real = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found.")
    
    # Use subset
    n_samples = min(1000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y_real = y_real[indices]
    
    # Random labels
    np.random.seed(42)
    y_random = np.random.permutation(y_real.copy())
    
    results = {}
    
    # Test with REAL labels
    X_train, X_test, y_train_real, y_test_real = train_test_split(
        X, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    X_train_final, X_val, y_train_final, y_val_final = train_test_split(
        X_train, y_train_real, test_size=0.2, random_state=42, stratify=y_train_real
    )
    
    detector_real = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_real.train(
        X_train_final, y_train_final,
        X_val=X_val, y_val=y_val_final,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred_real = detector_real.predict(X_test)
    auc_real = roc_auc_score(y_test_real, y_pred_real)
    results['real'] = auc_real
    
    # Test with RANDOM labels
    X_train, X_test, y_train_rand, y_test_rand = train_test_split(
        X, y_random, test_size=0.2, random_state=42, stratify=y_random
    )
    X_train_final, X_val, y_train_final, y_val_final = train_test_split(
        X_train, y_train_rand, test_size=0.2, random_state=42, stratify=y_train_rand
    )
    
    detector_rand = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_rand.train(
        X_train_final, y_train_final,
        X_val=X_val, y_val=y_val_final,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred_rand = detector_rand.predict(X_test)
    auc_random = roc_auc_score(y_test_rand, y_pred_rand)
    results['random'] = auc_random
    
    print(f"\n  📊 Comparison Results:")
    print(f"     Real labels:   AUC = {auc_real:.4f}")
    print(f"     Random labels: AUC = {auc_random:.4f}")
    print(f"     Difference:    {auc_real - auc_random:.4f}")
    
    # Real should be significantly better than random
    assert auc_real > auc_random + 0.2, \
        f"Real labels should perform much better than random. " \
        f"Real={auc_real:.4f}, Random={auc_random:.4f}"
    
    # Random should be close to 0.5
    assert auc_random < 0.60, \
        f"Random labels should give AUC ≈ 0.5, got {auc_random:.4f}"
    
    print(f"  ✅ PASS: Real labels ({auc_real:.4f}) >> Random ({auc_random:.4f})")

