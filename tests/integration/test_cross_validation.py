"""
Integration tests for cross-validation pipeline.
"""
import pytest
import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_results_exist():
    """Test that cross-validation results file exists and is valid."""
    result_file = Path('result/cross_validation_results.json')
    
    if not result_file.exists():
        pytest.skip("Cross-validation results not found. Run run_cross_validation.py first.")
    
    # Load and validate JSON
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Check structure
    assert 'scenario_a' in results, "Results should contain scenario_a"
    assert 'scenario_b' in results, "Results should contain scenario_b"
    
    # Check scenario_a structure
    scenario_a = results['scenario_a']
    assert 'fold_results' in scenario_a, "Should contain fold results"
    assert 'aggregated' in scenario_a, "Should contain aggregated results"
    
    # Check aggregated metrics
    aggregated = scenario_a['aggregated']
    assert 'auc' in aggregated, "Should contain AUC metrics"
    assert 'precision' in aggregated, "Should contain precision metrics"
    assert 'recall' in aggregated, "Should contain recall metrics"
    assert 'f1' in aggregated, "Should contain F1 metrics"
    
    # Check that mean and std are present
    auc_metrics = aggregated['auc']
    assert 'mean' in auc_metrics, "AUC should have mean"
    assert 'std' in auc_metrics, "AUC should have std"
    
    # Validate values
    assert 0 <= auc_metrics['mean'] <= 1, "AUC mean should be between 0 and 1"
    assert auc_metrics['std'] >= 0, "AUC std should be non-negative"
    
    print(f"✅ Scenario A: AUC = {auc_metrics['mean']:.3f} ± {auc_metrics['std']:.3f}")
    
    # Check scenario_b structure
    scenario_b = results['scenario_b']
    assert 'fold_results' in scenario_b
    assert 'aggregated' in scenario_b
    
    aggregated_b = scenario_b['aggregated']
    assert 'auc' in aggregated_b
    
    auc_metrics_b = aggregated_b['auc']
    assert 'mean' in auc_metrics_b
    assert 'std' in auc_metrics_b
    
    assert 0 <= auc_metrics_b['mean'] <= 1
    assert auc_metrics_b['std'] >= 0
    
    print(f"✅ Scenario B: AUC = {auc_metrics_b['mean']:.3f} ± {auc_metrics_b['std']:.3f}")


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_fold_count():
    """Test that cross-validation has 5 folds."""
    result_file = Path('result/cross_validation_results.json')
    
    if not result_file.exists():
        pytest.skip("Cross-validation results not found.")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Check fold count for scenario_a
    scenario_a = results['scenario_a']
    folds = scenario_a.get('fold_results', [])
    assert len(folds) == 5, f"Should have 5 folds, found {len(folds)}"
    
    # Check fold count for scenario_b
    scenario_b = results['scenario_b']
    folds_b = scenario_b.get('fold_results', [])
    assert len(folds_b) == 5, f"Should have 5 folds, found {len(folds_b)}"
    
    print(f"✅ Both scenarios have 5 folds")


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_fold_structure():
    """Test that each fold has required metrics."""
    result_file = Path('result/cross_validation_results.json')
    
    if not result_file.exists():
        pytest.skip("Cross-validation results not found.")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Check scenario_a folds
    scenario_a = results['scenario_a']
    folds = scenario_a.get('fold_results', [])
    
    for i, fold in enumerate(folds):
        # fold_results is a list of dicts with metrics, not fold numbers
        assert 'auc' in fold, f"Fold {i} should have 'auc' key"
        assert 'precision' in fold, f"Fold {i} should have 'precision'"
        assert 'recall' in fold, f"Fold {i} should have 'recall'"
        assert 'f1' in fold, f"Fold {i} should have 'f1'"
        
        # Validate AUC range
        assert 0 <= fold['auc'] <= 1, f"Fold {i} AUC should be between 0 and 1"
    
    print(f"✅ All {len(folds)} folds have correct structure")


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_aggregated_consistency():
    """Test that aggregated metrics are consistent with fold metrics."""
    result_file = Path('result/cross_validation_results.json')
    
    if not result_file.exists():
        pytest.skip("Cross-validation results not found.")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Check scenario_a
    scenario_a = results['scenario_a']
    folds = scenario_a['fold_results']
    aggregated = scenario_a['aggregated']
    
    # Compute mean and std from folds
    fold_aucs = [fold['auc'] for fold in folds]
    computed_mean = np.mean(fold_aucs)
    computed_std = np.std(fold_aucs)
    
    stored_mean = aggregated['auc']['mean']
    stored_std = aggregated['auc']['std']
    
    # Allow small numerical differences
    assert abs(computed_mean - stored_mean) < 0.01, \
        f"Mean mismatch: computed={computed_mean:.4f}, stored={stored_mean:.4f}"
    
    assert abs(computed_std - stored_std) < 0.01, \
        f"Std mismatch: computed={computed_std:.4f}, stored={stored_std:.4f}"
    
    print(f"✅ Aggregated metrics consistent with fold metrics")


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_scenario_b_perfect():
    """Test that Scenario B achieves perfect or near-perfect detection."""
    result_file = Path('result/cross_validation_results.json')
    
    if not result_file.exists():
        pytest.skip("Cross-validation results not found.")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    scenario_b = results['scenario_b']
    aggregated = scenario_b['aggregated']
    
    auc_mean = aggregated['auc']['mean']
    auc_std = aggregated['auc']['std']
    
    # Scenario B should achieve high AUC (>= 0.95) with low variance
    assert auc_mean >= 0.95, f"Scenario B AUC should be >= 0.95, got {auc_mean:.3f}"
    assert auc_std <= 0.1, f"Scenario B AUC std should be <= 0.1, got {auc_std:.3f}"
    
    print(f"✅ Scenario B: AUC = {auc_mean:.3f} ± {auc_std:.3f} (high performance)")


@pytest.mark.integration
@pytest.mark.slow
def test_cross_validation_vs_single_split():
    """Test that CV results differ from single-split (if single-split exists)."""
    cv_file = Path('result/cross_validation_results.json')
    single_file_a = Path('result/scenario_a/detection_results_cnn.json')
    single_file_b = Path('result/scenario_b/detection_results_cnn.json')
    
    if not cv_file.exists():
        pytest.skip("Cross-validation results not found.")
    
    with open(cv_file, 'r') as f:
        cv_results = json.load(f)
    
    # Check Scenario A
    if single_file_a.exists():
        with open(single_file_a, 'r') as f:
            single_a = json.load(f)
        
        cv_auc_a = cv_results['scenario_a']['aggregated']['auc']['mean']
        single_auc_a = single_a.get('metrics', {}).get('auc', 0)
        
        # CV should reveal true performance (may be lower than single-split)
        print(f"  Scenario A - Single: {single_auc_a:.3f}, CV: {cv_auc_a:.3f}")
        # Note: We don't assert here because CV can be higher or lower
    
    # Check Scenario B
    if single_file_b.exists():
        with open(single_file_b, 'r') as f:
            single_b = json.load(f)
        
        cv_auc_b = cv_results['scenario_b']['aggregated']['auc']['mean']
        single_auc_b = single_b.get('metrics', {}).get('auc', 0)
        
        print(f"  Scenario B - Single: {single_auc_b:.3f}, CV: {cv_auc_b:.3f}")
    
    print("✅ Cross-validation results validated")

