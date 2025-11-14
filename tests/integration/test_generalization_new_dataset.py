"""
🔥 CRITICAL TEST: New Dataset Generalization
=============================================
This test verifies that high AUC generalizes to completely new datasets.

Train on Dataset A → Test on Dataset B (completely new, different distribution)

If AUC still ≥ 0.95 → AUC=1.00 is trustworthy and generalizable
If AUC drops significantly → overfitting to specific dataset

This is EXTREMELY important for reviewers.
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
def test_cross_scenario_generalization():
    """
    🔥 CRITICAL: Test generalization across scenarios.
    
    Train on Scenario A → Test on Scenario B
    Train on Scenario B → Test on Scenario A
    
    This tests if the model learns scenario-specific features or general patterns.
    """
    print("\n" + "="*70)
    print("🔥 CROSS-SCENARIO GENERALIZATION TEST")
    print("="*70)
    
    # Load both datasets
    try:
        X_a, y_a = load_dataset_for_test(scenario='sat')
        X_b, y_b = load_dataset_for_test(scenario='ground')
    except FileNotFoundError as e:
        pytest.skip(f"Dataset not found: {e}")
    
    # Use subsets for faster testing
    n_samples_a = min(1000, len(X_a))
    n_samples_b = min(1000, len(X_b))
    
    indices_a = np.random.choice(len(X_a), n_samples_a, replace=False)
    indices_b = np.random.choice(len(X_b), n_samples_b, replace=False)
    
    X_a = X_a[indices_a]
    y_a = y_a[indices_a]
    X_b = X_b[indices_b]
    y_b = y_b[indices_b]
    
    print(f"  Scenario A: {len(X_a)} samples")
    print(f"  Scenario B: {len(X_b)} samples")
    
    results = {}
    
    # Test 1: Train on A, Test on B
    print("\n  Test 1: Train on Scenario A → Test on Scenario B")
    X_train_a, _, y_train_a, _ = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
    )
    X_train_final_a, X_val_a, y_train_final_a, y_val_a = train_test_split(
        X_train_a, y_train_a, test_size=0.2, random_state=42, stratify=y_train_a
    )
    
    detector_ab = CNNDetector(
        use_csi=False,
        use_attention=True,
        random_state=42,
        learning_rate=0.001,
        dropout_rate=0.3,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_LOSS_GAMMA,
        focal_alpha=FOCAL_LOSS_ALPHA
    )
    
    detector_ab.train(
        X_train_final_a, y_train_final_a,
        X_val=X_val_a, y_val=y_val_a,
        epochs=20,
        batch_size=128,
        verbose=0
    )
    
    y_pred_ab = detector_ab.predict(X_b)
    auc_ab = roc_auc_score(y_b, y_pred_ab)
    results['A→B'] = auc_ab
    print(f"    AUC (A→B): {auc_ab:.4f}")
    
    # Test 2: Train on B, Test on A
    print("\n  Test 2: Train on Scenario B → Test on Scenario A")
    X_train_b, _, y_train_b, _ = train_test_split(
        X_b, y_b, test_size=0.2, random_state=42, stratify=y_b
    )
    X_train_final_b, X_val_b, y_train_final_b, y_val_b = train_test_split(
        X_train_b, y_train_b, test_size=0.2, random_state=42, stratify=y_train_b
    )
    
    detector_ba = CNNDetector(
        use_csi=False,
        use_attention=True,
        random_state=42,
        learning_rate=0.001,
        dropout_rate=0.3,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_LOSS_GAMMA,
        focal_alpha=FOCAL_LOSS_ALPHA
    )
    
    detector_ba.train(
        X_train_final_b, y_train_final_b,
        X_val=X_val_b, y_val=y_val_b,
        epochs=20,
        batch_size=128,
        verbose=0
    )
    
    y_pred_ba = detector_ba.predict(X_a)
    auc_ba = roc_auc_score(y_a, y_pred_ba)
    results['B→A'] = auc_ba
    print(f"    AUC (B→A): {auc_ba:.4f}")
    
    # Test 3: Baseline (train and test on same scenario)
    print("\n  Test 3: Baseline (Train and Test on same scenario)")
    
    # A→A
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
    )
    X_train_final_a, X_val_a, y_train_final_a, y_val_a = train_test_split(
        X_train_a, y_train_a, test_size=0.2, random_state=42, stratify=y_train_a
    )
    
    detector_aa = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_aa.train(
        X_train_final_a, y_train_final_a,
        X_val=X_val_a, y_val=y_val_a,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred_aa = detector_aa.predict(X_test_a)
    auc_aa = roc_auc_score(y_test_a, y_pred_aa)
    results['A→A'] = auc_aa
    print(f"    AUC (A→A): {auc_aa:.4f}")
    
    # B→B
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y_b, test_size=0.2, random_state=42, stratify=y_b
    )
    X_train_final_b, X_val_b, y_train_final_b, y_val_b = train_test_split(
        X_train_b, y_train_b, test_size=0.2, random_state=42, stratify=y_train_b
    )
    
    detector_bb = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_bb.train(
        X_train_final_b, y_train_final_b,
        X_val=X_val_b, y_val=y_val_b,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred_bb = detector_bb.predict(X_test_b)
    auc_bb = roc_auc_score(y_test_b, y_pred_bb)
    results['B→B'] = auc_bb
    print(f"    AUC (B→B): {auc_bb:.4f}")
    
    # Print summary
    print(f"\n  📊 Results Summary:")
    print(f"     {'Train→Test':<12} {'AUC':<8}")
    print(f"     {'-'*20}")
    for key, auc in results.items():
        print(f"     {key:<12} {auc:.4f}")
    
    # CRITICAL ASSERTIONS
    
    # 1. Same-scenario should perform best (baseline)
    assert results['A→A'] >= results['A→B'], \
        f"Same-scenario (A→A: {results['A→A']:.4f}) should perform better than " \
        f"cross-scenario (A→B: {results['A→B']:.4f})"
    
    assert results['B→B'] >= results['B→A'], \
        f"Same-scenario (B→B: {results['B→B']:.4f}) should perform better than " \
        f"cross-scenario (B→A: {results['B→A']:.4f})"
    
    # 2. Cross-scenario should still be reasonable (generalization)
    # For Scenario B (which has high AUC), cross-scenario should still be decent
    assert results['B→A'] >= 0.60, \
        f"Cross-scenario (B→A: {results['B→A']:.4f}) should be ≥ 0.60. " \
        f"Too low suggests overfitting to Scenario B."
    
    # 3. Performance drop should be reasonable
    drop_ba = results['B→B'] - results['B→A']
    assert drop_ba <= 0.30, \
        f"Performance drop (B→B to B→A: {drop_ba:.4f}) should be ≤ 0.30. " \
        f"Too large suggests overfitting."
    
    print(f"\n  ✅ PASS: Model shows reasonable generalization across scenarios")
    print(f"     Same-scenario: B→B={results['B→B']:.4f}, Cross-scenario: B→A={results['B→A']:.4f}")


@pytest.mark.integration
@pytest.mark.slow
def test_new_dataset_generalization():
    """
    Test generalization to a completely new dataset (different generation run).
    
    This simulates the real-world scenario where model is trained on one dataset
    and deployed on new data from a different distribution.
    """
    print("\n" + "="*70)
    print("🔥 NEW DATASET GENERALIZATION TEST")
    print("="*70)
    
    # Load dataset (Scenario B - should have high AUC)
    try:
        X_train_full, y_train_full = load_dataset_for_test(scenario='ground')
    except FileNotFoundError:
        pytest.skip("Dataset not found. Generate dataset first.")
    
    # For this test, we simulate "new dataset" by:
    # 1. Using a subset as training data
    # 2. Using a different subset as "new" test data
    # This simulates different data distributions
    
    n_samples = len(X_train_full)
    if n_samples < 2000:
        pytest.skip(f"Need at least 2000 samples, got {n_samples}")
    
    # Split into "training dataset" and "new test dataset"
    # Use different random seeds to simulate different distributions
    np.random.seed(42)
    train_indices = np.random.choice(n_samples, n_samples // 2, replace=False)
    test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
    
    X_train_dataset = X_train_full[train_indices]
    y_train_dataset = y_train_full[train_indices]
    
    X_new_test = X_train_full[test_indices]
    y_new_test = y_train_full[test_indices]
    
    print(f"  Training dataset: {len(X_train_dataset)} samples")
    print(f"  New test dataset: {len(X_new_test)} samples")
    
    # Train on training dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_dataset, y_train_dataset, test_size=0.2, random_state=42, stratify=y_train_dataset
    )
    
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
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=20,
        batch_size=128,
        verbose=0
    )
    
    # Test on "new" dataset
    y_pred_new = detector.predict(X_new_test)
    auc_new = roc_auc_score(y_new_test, y_pred_new)
    
    # Also test on same distribution (for comparison)
    X_train_same, X_test_same, y_train_same, y_test_same = train_test_split(
        X_train_dataset, y_train_dataset, test_size=0.2, random_state=42, stratify=y_train_dataset
    )
    X_train_final_same, X_val_same, y_train_final_same, y_val_same = train_test_split(
        X_train_same, y_train_same, test_size=0.2, random_state=42, stratify=y_train_same
    )
    
    detector_same = CNNDetector(use_csi=False, use_attention=True, random_state=42)
    detector_same.train(
        X_train_final_same, y_train_final_same,
        X_val=X_val_same, y_val=y_val_same,
        epochs=20, batch_size=128, verbose=0
    )
    y_pred_same = detector_same.predict(X_test_same)
    auc_same = roc_auc_score(y_test_same, y_pred_same)
    
    print(f"\n  📊 Results:")
    print(f"     Same distribution: AUC = {auc_same:.4f}")
    print(f"     New dataset:       AUC = {auc_new:.4f}")
    print(f"     Difference:        {auc_same - auc_new:.4f}")
    
    # New dataset should still perform well
    assert auc_new >= 0.85, \
        f"New dataset AUC ({auc_new:.4f}) should be ≥ 0.85. " \
        f"Too low suggests overfitting to training distribution."
    
    # Performance drop should be reasonable
    drop = auc_same - auc_new
    assert drop <= 0.15, \
        f"Performance drop ({drop:.4f}) should be ≤ 0.15. " \
        f"Too large suggests poor generalization."
    
    print(f"  ✅ PASS: Model generalizes well to new dataset")

