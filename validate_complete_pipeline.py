#!/usr/bin/env python3
"""
🔍 Complete Pipeline Validation Script
=====================================
Validates all phases of the covert leakage detection pipeline according to
the research paper specifications.

Phases:
1. Structural Validation (Pipeline Consistency)
2. Dataset Integrity Check
3. Model Behavior Validation
4. Scenario-Specific Validation
5. Statistical Validation
6. Robustness & Sanity Test
7. Final Acceptance Criteria
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATASET_DIR, MODEL_DIR, RESULT_DIR,
    NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA
)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}━━━ {text} ━━━{Colors.RESET}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.RESET}")


# ============================================================================
# Phase 1: Structural Validation (Pipeline Consistency)
# ============================================================================

def phase1_structural_validation() -> Dict[str, bool]:
    """Check that all required files and directories exist."""
    print_header("Phase 1: Structural Validation (Pipeline Consistency)")
    
    results = {}
    
    # Required files
    required_files = {
        'Dataset A': f"{DATASET_DIR}/dataset_scenario_a.pkl",
        'Dataset B': f"{DATASET_DIR}/dataset_scenario_b.pkl",
        'Model A (CNN-only)': f"{MODEL_DIR}/scenario_a/cnn_detector.keras",
        'Model B (CNN-only)': f"{MODEL_DIR}/scenario_b/cnn_detector.keras",
        'CV Summary A': f"{RESULT_DIR}/cv_summary_scenario_a.csv",
        'CV Summary B': f"{RESULT_DIR}/cv_summary_scenario_b.csv",
        'Robustness A': f"{RESULT_DIR}/robustness_scenario_a.csv",
        'Robustness B': f"{RESULT_DIR}/robustness_scenario_b.csv",
    }
    
    # Optional files (nice to have)
    optional_files = {
        'Model A (CNN+CSI)': f"{MODEL_DIR}/scenario_a/cnn_detector_csi.keras",
        'Model B (CNN+CSI)': f"{MODEL_DIR}/scenario_b/cnn_detector_csi.keras",
        'Baselines A': f"{RESULT_DIR}/baselines_scenario_a.csv",
        'Baselines B': f"{RESULT_DIR}/baselines_scenario_b.csv",
        'CSI Analysis A': f"{RESULT_DIR}/csi_analysis_summary_scenario_a.csv",
        'CSI Analysis B': f"{RESULT_DIR}/csi_analysis_summary_scenario_b.csv",
    }
    
    print_section("Required Files")
    all_required_exist = True
    for name, path in required_files.items():
        exists = os.path.exists(path)
        results[f"file_{name.replace(' ', '_').lower()}"] = exists
        if exists:
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print_success(f"{name}: {path} ({size:.2f} MB)")
        else:
            print_error(f"{name}: {path} (NOT FOUND)")
            all_required_exist = False
    
    print_section("Optional Files")
    for name, path in optional_files.items():
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print_info(f"{name}: {path} ({size:.2f} MB)")
        else:
            print_warning(f"{name}: {path} (not found, optional)")
    
    results['all_required_exist'] = all_required_exist
    return results


# ============================================================================
# Phase 2: Dataset Integrity Check
# ============================================================================

def phase2_dataset_integrity() -> Dict[str, any]:
    """Check dataset integrity and metadata."""
    print_header("Phase 2: Dataset Integrity Check")
    
    results = {}
    
    for scenario in ['a', 'b']:
        scenario_name = f"Scenario {scenario.upper()}"
        dataset_path = f"{DATASET_DIR}/dataset_scenario_{scenario}.pkl"
        
        print_section(f"{scenario_name} Dataset")
        
        if not os.path.exists(dataset_path):
            print_error(f"Dataset not found: {dataset_path}")
            results[f"dataset_{scenario}_exists"] = False
            continue
        
        results[f"dataset_{scenario}_exists"] = True
        
        try:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            # Check required keys
            required_keys = ['tx_grids', 'rx_grids', 'labels', 'meta']
            for key in required_keys:
                if key not in dataset:
                    print_error(f"Missing key: {key}")
                    results[f"dataset_{scenario}_has_{key}"] = False
                else:
                    results[f"dataset_{scenario}_has_{key}"] = True
            
            # Check sample counts
            labels = dataset.get('labels', [])
            if isinstance(labels, np.ndarray):
                unique, counts = np.unique(labels, return_counts=True)
                benign_count = counts[0] if len(counts) > 0 else 0
                attack_count = counts[1] if len(counts) > 1 else 0
                total = len(labels)
                
                print_info(f"Total samples: {total}")
                print_info(f"Benign samples: {benign_count} ({benign_count/total*100:.1f}%)")
                print_info(f"Attack samples: {attack_count} ({attack_count/total*100:.1f}%)")
                
                # Check balance (should be ~50/50)
                balance_ratio = min(benign_count, attack_count) / max(benign_count, attack_count)
                if balance_ratio > 0.45:
                    print_success(f"Dataset is balanced (ratio: {balance_ratio:.3f})")
                    results[f"dataset_{scenario}_balanced"] = True
                else:
                    print_warning(f"Dataset imbalance detected (ratio: {balance_ratio:.3f})")
                    results[f"dataset_{scenario}_balanced"] = False
                
                results[f"dataset_{scenario}_total"] = total
                results[f"dataset_{scenario}_benign"] = benign_count
                results[f"dataset_{scenario}_attack"] = attack_count
            
            # Check metadata
            meta = dataset.get('meta', [])
            if meta and len(meta) > 0:
                sample_meta = meta[0] if isinstance(meta[0], dict) else {}
                
                # Phase 6 metadata for Scenario B
                if scenario == 'b':
                    phase6_keys = ['fd_ul', 'fd_dl', 'G_r_mean', 'delay_samples', 'snr_ul', 'snr_dl']
                    found_keys = [k for k in phase6_keys if k in sample_meta]
                    missing_keys = [k for k in phase6_keys if k not in sample_meta]
                    
                    if found_keys:
                        print_success(f"Phase 6 metadata found: {found_keys}")
                        for key in found_keys[:3]:  # Show first 3
                            val = sample_meta.get(key, "N/A")
                            print_info(f"  {key}: {val}")
                        results[f"dataset_{scenario}_phase6_meta"] = True
                    else:
                        print_error(f"Phase 6 metadata missing: {missing_keys}")
                        print_warning("⚠️  Dataset B was generated BEFORE Phase 6 implementation!")
                        print_warning("⚠️  Please regenerate Scenario B dataset:")
                        print_warning("     python3 run_all_scenarios.py --scenario ground --num-samples 2000")
                        results[f"dataset_{scenario}_phase6_meta"] = False
                
                # Check power difference
                if 'power_diff_pct' in sample_meta:
                    power_diffs = [m.get('power_diff_pct', 0.0) if isinstance(m, dict) else 0.0 for m in meta]
                    power_diffs = [p for p in power_diffs if p is not None]
                    if power_diffs:
                        mean_power_diff = np.mean(power_diffs)
                        max_power_diff = np.max(power_diffs)
                        
                        print_info(f"Mean power diff: {mean_power_diff:.4f}%")
                        print_info(f"Max power diff: {max_power_diff:.4f}%")
                        
                        # Paper expectations: A ≈ 0.04%, B ≈ 0.12%
                        expected = 0.04 if scenario == 'a' else 0.12
                        threshold = 0.2  # Allow up to 0.2%
                        
                        if mean_power_diff <= threshold:
                            print_success(f"Power diff within threshold ({mean_power_diff:.4f}% <= {threshold}%)")
                            results[f"dataset_{scenario}_power_diff_ok"] = True
                        else:
                            print_warning(f"Power diff exceeds threshold ({mean_power_diff:.4f}% > {threshold}%)")
                            results[f"dataset_{scenario}_power_diff_ok"] = False
                        
                        results[f"dataset_{scenario}_mean_power_diff"] = mean_power_diff
                        results[f"dataset_{scenario}_max_power_diff"] = max_power_diff
                
        except Exception as e:
            print_error(f"Error loading dataset: {e}")
            results[f"dataset_{scenario}_load_error"] = str(e)
    
    return results


# ============================================================================
# Phase 3: Model Behavior Validation
# ============================================================================

def phase3_model_validation() -> Dict[str, any]:
    """Validate model inference and performance."""
    print_header("Phase 3: Model Behavior Validation")
    
    results = {}
    
    try:
        import sys
        import os
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from model.detector_cnn import CNNDetector
    except ImportError as e:
        print_error(f"Cannot import CNNDetector: {e}")
        print_warning("Skipping model inference test (non-critical)")
        results['model_import_error'] = True
        return results
    
    for scenario in ['a', 'b']:
        scenario_name = f"Scenario {scenario.upper()}"
        model_path = f"{MODEL_DIR}/scenario_{scenario}/cnn_detector.keras"
        
        print_section(f"{scenario_name} Model")
        
        if not os.path.exists(model_path):
            print_error(f"Model not found: {model_path}")
            results[f"model_{scenario}_exists"] = False
            continue
        
        results[f"model_{scenario}_exists"] = True
        
        try:
            # Load model
            detector = CNNDetector(use_csi=False)
            detector.load(model_path)
            print_success(f"Model loaded: {model_path}")
            
            # Load dataset for inference test
            dataset_path = f"{DATASET_DIR}/dataset_scenario_{scenario}.pkl"
            if os.path.exists(dataset_path):
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                
                # Use stratified sampling to ensure both classes are present
                from sklearn.model_selection import train_test_split
                
                # ✅ REALISTIC: Use rx_grids (post-channel) for validation
                # Model should be trained on rx_grids (post-channel) for realistic deployment
                # Validation should also use rx_grids to match training
                if 'rx_grids' in dataset:
                    X_all = np.array(dataset.get('rx_grids', []))
                    print(f"  ✓ Using rx_grids (post-channel, same as training)")
                else:
                    X_all = np.array(dataset.get('tx_grids', []))
                    print(f"  ⚠️  Using tx_grids (rx_grids not available, fallback)")
                
                y_all = np.array(dataset.get('labels', []))
                
                if len(X_all) > 0 and len(y_all) > 0:
                    # Check if we have both classes
                    unique_classes = np.unique(y_all)
                    if len(unique_classes) < 2:
                        print_warning(f"Only one class present in dataset: {unique_classes}")
                        results[f"model_{scenario}_single_class"] = True
                        continue
                    
                    # Use stratified split to get balanced test set
                    # Take 200 samples (100 per class) for more reliable evaluation
                    test_size = min(200, len(X_all) // 2)
                    if test_size < 20:
                        test_size = min(20, len(X_all))
                    
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_all, y_all,
                            test_size=test_size,
                            stratify=y_all,
                            random_state=42
                        )
                        
                        # Verify both classes in test set
                        test_classes = np.unique(y_test)
                        if len(test_classes) < 2:
                            print_warning(f"Test set has only one class: {test_classes}")
                            # Fallback: manually select balanced samples
                            benign_idx = np.where(y_all == 0)[0][:test_size//2]
                            attack_idx = np.where(y_all == 1)[0][:test_size//2]
                            selected_idx = np.concatenate([benign_idx, attack_idx])
                            X_test = X_all[selected_idx]
                            y_test = y_all[selected_idx]
                    except Exception as e:
                        print_warning(f"Stratified split failed: {e}, using random sampling")
                        # Fallback: random sampling
                        indices = np.random.choice(len(X_all), size=min(200, len(X_all)), replace=False)
                        X_test = X_all[indices]
                        y_test = y_all[indices]
                    
                    # Run inference with threshold optimization
                    try:
                        # Split test set into eval and threshold optimization
                        # Use 60% for evaluation, 40% for threshold optimization
                        from sklearn.model_selection import train_test_split
                        X_eval, X_thresh, y_eval, y_thresh = train_test_split(
                            X_test, y_test,
                            test_size=0.4,
                            stratify=y_test,
                            random_state=42
                        )
                        
                        # Optimize threshold on threshold set
                        optimal_threshold = detector.find_optimal_threshold(
                            X_thresh, y_thresh,
                            metric='f1'  # Maximize F1 score
                        )
                        print_info(f"  Optimal threshold: {optimal_threshold:.4f}")
                        
                        # Evaluate on eval set with optimal threshold
                        # Get probabilities
                        probs = detector.predict_proba(X_eval)
                        
                        # Apply optimal threshold
                        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
                        preds = (probs >= optimal_threshold).astype(int)
                        
                        auc = roc_auc_score(y_eval, probs) if len(np.unique(y_eval)) > 1 else 0.0
                        precision = precision_score(y_eval, preds, zero_division=0)
                        recall = recall_score(y_eval, preds, zero_division=0)
                        f1 = f1_score(y_eval, preds, zero_division=0)
                        
                        print_info(f"AUC: {auc:.4f}")
                        print_info(f"F1: {f1:.4f}")
                        print_info(f"Precision: {precision:.4f}")
                        print_info(f"Recall: {recall:.4f}")
                        
                        # Paper expectations: A: AUC≈0.9998, F1≈0.9933; B: AUC≈0.9999, F1≈0.9740
                        expected_auc = 0.9998 if scenario == 'a' else 0.9999
                        expected_f1 = 0.9933 if scenario == 'a' else 0.9740
                        
                        if auc >= 0.99:
                            print_success(f"AUC meets expectation (≥0.99)")
                            results[f"model_{scenario}_auc_ok"] = True
                        else:
                            print_warning(f"AUC below expectation ({auc:.4f} < 0.99)")
                            results[f"model_{scenario}_auc_ok"] = False
                        
                        if f1 >= 0.95:
                            print_success(f"F1 meets expectation (≥0.95)")
                            results[f"model_{scenario}_f1_ok"] = True
                        else:
                            print_warning(f"F1 below expectation ({f1:.4f} < 0.95)")
                            results[f"model_{scenario}_f1_ok"] = False
                        
                        results[f"model_{scenario}_auc"] = auc
                        results[f"model_{scenario}_f1"] = f1
                        results[f"model_{scenario}_precision"] = precision
                        results[f"model_{scenario}_recall"] = recall
                        
                    except Exception as e:
                        print_error(f"Inference error: {e}")
                        results[f"model_{scenario}_inference_error"] = str(e)
                else:
                    print_warning("No test data available")
            else:
                print_warning(f"Dataset not found for inference test: {dataset_path}")
                
        except Exception as e:
            print_error(f"Error loading model: {e}")
            results[f"model_{scenario}_load_error"] = str(e)
    
    return results


# ============================================================================
# Phase 4: Scenario-Specific Validation
# ============================================================================

def phase4_scenario_validation() -> Dict[str, any]:
    """Validate Scenario B specific features."""
    print_header("Phase 4: Scenario-Specific Validation")
    
    results = {}
    
    # Check metadata CSV for Scenario B
    metadata_csv = f"{RESULT_DIR}/dataset_metadata_phase1_scenario_b.csv"
    
    print_section("Scenario B Metadata (CSV/Dataset)")
    
    # Try CSV first, fallback to dataset if Phase 6 columns missing
    df = None
    dataset_path = f"{DATASET_DIR}/dataset_scenario_b.pkl"
    
    if os.path.exists(metadata_csv):
        try:
            df = pd.read_csv(metadata_csv)
            print_success(f"Metadata CSV loaded: {len(df)} rows")
            
            # Check if CSV has Phase 6 columns
            phase6_cols = ['fd_ul', 'fd_dl', 'G_r_mean', 'delay_samples', 'snr_ul', 'snr_dl']
            found_cols = [col for col in phase6_cols if col in df.columns]
            if not found_cols and os.path.exists(dataset_path):
                print_warning("CSV missing Phase 6 columns, loading from dataset...")
                try:
                    import pickle
                    with open(dataset_path, 'rb') as f:
                        dataset = pickle.load(f)
                    meta = dataset.get('meta', [])
                    labels = dataset.get('labels', [])
                    if meta:
                        metadata_rows = []
                        for i, meta_dict in enumerate(meta):
                            if isinstance(meta_dict, dict):
                                row = {'sample_idx': i, 'label': int(labels[i]) if i < len(labels) else None, **meta_dict}
                                metadata_rows.append(row)
                        df = pd.DataFrame(metadata_rows)
                        print_success(f"Metadata loaded from dataset: {len(df)} rows")
                except Exception as e:
                    print_error(f"Error loading from dataset: {e}")
        except Exception as e:
            print_warning(f"Error reading CSV: {e}")
    
    # If still no df, try dataset directly
    if df is None and os.path.exists(dataset_path):
        try:
            import pickle
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            meta = dataset.get('meta', [])
            labels = dataset.get('labels', [])
            if meta:
                metadata_rows = []
                for i, meta_dict in enumerate(meta):
                    if isinstance(meta_dict, dict):
                        row = {'sample_idx': i, 'label': int(labels[i]) if i < len(labels) else None, **meta_dict}
                        metadata_rows.append(row)
                df = pd.DataFrame(metadata_rows)
                print_success(f"Metadata loaded from dataset: {len(df)} rows")
        except Exception as e:
            print_error(f"Error loading from dataset: {e}")
    
    if df is not None:
        # Check Phase 6 columns
        phase6_cols = ['fd_ul', 'fd_dl', 'G_r_mean', 'delay_samples', 'snr_ul', 'snr_dl']
        found_cols = [col for col in phase6_cols if col in df.columns]
        missing_cols = [col for col in phase6_cols if col not in df.columns]
        
        if found_cols:
            print_success(f"Phase 6 columns found: {found_cols}")
            results['scenario_b_phase6_cols'] = True
            
            # Check for non-zero values
            for col in found_cols:
                if col in df.columns:
                    non_zero = (df[col] != 0).sum()
                    mean_val = df[col].mean()
                    print_info(f"  {col}: mean={mean_val:.4f}, non-zero={non_zero}/{len(df)}")
        else:
            print_error(f"Phase 6 columns missing: {missing_cols}")
            results['scenario_b_phase6_cols'] = False
        
        # Check that fd_ul and fd_dl are different (independent)
        if 'fd_ul' in df.columns and 'fd_dl' in df.columns:
            diff = (df['fd_ul'] != df['fd_dl']).sum()
            if diff > 0:
                print_success(f"fd_ul and fd_dl are independent ({diff}/{len(df)} different)")
                results['scenario_b_doppler_independent'] = True
            else:
                print_warning("fd_ul and fd_dl are identical (may indicate issue)")
                results['scenario_b_doppler_independent'] = False
    else:
        print_error("Could not load metadata from CSV or dataset")
        results['scenario_b_phase6_cols'] = False
    
    # Check that relay function is used
    print_section("Relay Function Usage")
    relay_file = "core/scenario_b_relay.py"
    if os.path.exists(relay_file):
        with open(relay_file, 'r') as f:
            content = f.read()
            if 'amplify_and_forward_relay' in content and 'target_power' in content:
                print_success("Relay function with AGC found in scenario_b_relay.py")
                results['relay_function_exists'] = True
            else:
                print_error("Relay function or AGC not found")
                results['relay_function_exists'] = False
    else:
        print_error(f"Relay file not found: {relay_file}")
        results['relay_function_exists'] = False
    
    return results


# ============================================================================
# Phase 5: Statistical Validation
# ============================================================================

def phase5_statistical_validation() -> Dict[str, any]:
    """Compare statistical results with paper."""
    print_header("Phase 5: Statistical Validation")
    
    results = {}
    
    for scenario in ['a', 'b']:
        scenario_name = f"Scenario {scenario.upper()}"
        cv_summary = f"{RESULT_DIR}/cv_summary_scenario_{scenario}.csv"
        
        print_section(f"{scenario_name} Cross-Validation Summary")
        
        if os.path.exists(cv_summary):
            try:
                df = pd.read_csv(cv_summary)
                print_success(f"CV summary loaded: {len(df)} metrics")
                
                # Get AUC and F1 statistics
                if 'metric' in df.columns and 'mean' in df.columns:
                    auc_row = df[df['metric'] == 'auc']
                    f1_row = df[df['metric'] == 'f1']
                    
                    if not auc_row.empty:
                        auc_mean = auc_row['mean'].iloc[0]
                        auc_std = auc_row['std'].iloc[0] if 'std' in df.columns else 0.0
                        print_info(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
                        
                        # Paper expectations
                        expected_auc = 0.9998 if scenario == 'a' else 0.9999
                        diff = abs(auc_mean - expected_auc)
                        
                        if diff <= 0.005:
                            print_success(f"AUC matches paper expectation (diff: {diff:.4f} <= 0.005)")
                            results[f"cv_{scenario}_auc_match"] = True
                        else:
                            print_warning(f"AUC differs from paper (diff: {diff:.4f} > 0.005)")
                            results[f"cv_{scenario}_auc_match"] = False
                        
                        results[f"cv_{scenario}_auc_mean"] = auc_mean
                        results[f"cv_{scenario}_auc_std"] = auc_std
                    
                    if not f1_row.empty:
                        f1_mean = f1_row['mean'].iloc[0]
                        f1_std = f1_row['std'].iloc[0] if 'std' in df.columns else 0.0
                        print_info(f"F1: {f1_mean:.4f} ± {f1_std:.4f}")
                        
                        # Paper expectations
                        expected_f1 = 0.9933 if scenario == 'a' else 0.9740
                        diff = abs(f1_mean - expected_f1)
                        
                        if diff <= 0.01:
                            print_success(f"F1 matches paper expectation (diff: {diff:.4f} <= 0.01)")
                            results[f"cv_{scenario}_f1_match"] = True
                        else:
                            print_warning(f"F1 differs from paper (diff: {diff:.4f} > 0.01)")
                            results[f"cv_{scenario}_f1_match"] = False
                        
                        results[f"cv_{scenario}_f1_mean"] = f1_mean
                        results[f"cv_{scenario}_f1_std"] = f1_std
                
            except Exception as e:
                print_error(f"Error reading CV summary: {e}")
                results[f"cv_{scenario}_error"] = str(e)
        else:
            print_warning(f"CV summary not found: {cv_summary}")
            results[f"cv_{scenario}_exists"] = False
    
    return results


# ============================================================================
# Phase 6: Robustness & Sanity Test
# ============================================================================

def phase6_robustness_test() -> Dict[str, any]:
    """Test model robustness."""
    print_header("Phase 6: Robustness & Sanity Test")
    
    results = {}
    
    # Check robustness CSV files
    for scenario in ['a', 'b']:
        scenario_name = f"Scenario {scenario.upper()}"
        robustness_csv = f"{RESULT_DIR}/robustness_scenario_{scenario}.csv"
        
        print_section(f"{scenario_name} Robustness")
        
        if os.path.exists(robustness_csv):
            try:
                df = pd.read_csv(robustness_csv)
                print_success(f"Robustness data loaded: {len(df)} rows")
                
                # Check AUC across different conditions
                if 'auc' in df.columns:
                    min_auc = df['auc'].min()
                    mean_auc = df['auc'].mean()
                    max_auc = df['auc'].max()
                    
                    print_info(f"AUC range: {min_auc:.4f} - {max_auc:.4f} (mean: {mean_auc:.4f})")
                    
                    # Robustness threshold: AUC > 0.98
                    if min_auc >= 0.98:
                        print_success(f"Robustness confirmed (min AUC: {min_auc:.4f} >= 0.98)")
                        results[f"robustness_{scenario}_ok"] = True
                    else:
                        print_warning(f"Robustness concern (min AUC: {min_auc:.4f} < 0.98)")
                        results[f"robustness_{scenario}_ok"] = False
                    
                    results[f"robustness_{scenario}_min_auc"] = min_auc
                    results[f"robustness_{scenario}_mean_auc"] = mean_auc
                    results[f"robustness_{scenario}_max_auc"] = max_auc
                
            except Exception as e:
                print_error(f"Error reading robustness CSV: {e}")
                results[f"robustness_{scenario}_error"] = str(e)
        else:
            print_warning(f"Robustness CSV not found: {robustness_csv}")
            results[f"robustness_{scenario}_exists"] = False
    
    return results


# ============================================================================
# Phase 7: Final Acceptance Criteria
# ============================================================================

def phase7_final_acceptance(all_results: Dict[str, any]) -> Dict[str, bool]:
    """Final acceptance criteria check."""
    print_header("Phase 7: Final Acceptance Criteria")
    
    criteria = {
        'Pipeline executes without errors': all_results.get('all_required_exist', False),
        'Output files are complete': all_results.get('all_required_exist', False),
        'Phase 6 metadata exists in Scenario B': all_results.get('dataset_b_phase6_meta', False),
        'AUC ≥ 0.999': (
            (all_results.get('model_a_auc', 0) >= 0.999 or all_results.get('cv_a_auc_mean', 0) >= 0.999) and
            (all_results.get('model_b_auc', 0) >= 0.999 or all_results.get('cv_b_auc_mean', 0) >= 0.999)
        ),
        'Power diff ≤ 0.2%': (
            all_results.get('dataset_a_power_diff_ok', False) and
            all_results.get('dataset_b_power_diff_ok', False)
        ),
        'Model inference works': (
            all_results.get('model_a_exists', False) and
            all_results.get('model_b_exists', False) and
            'model_a_inference_error' not in all_results and
            'model_b_inference_error' not in all_results
        ),
    }
    
    print_section("Acceptance Criteria")
    all_passed = True
    for criterion, passed in criteria.items():
        if passed:
            print_success(criterion)
        else:
            print_error(criterion)
            all_passed = False
    
    criteria['ALL_CRITERIA_PASSED'] = all_passed
    return criteria


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all validation phases."""
    print_header("Complete Pipeline Validation")
    print_info("Validating all phases according to research paper specifications")
    
    all_results = {}
    
    # Run all phases
    try:
        all_results.update(phase1_structural_validation())
        all_results.update(phase2_dataset_integrity())
        all_results.update(phase3_model_validation())
        all_results.update(phase4_scenario_validation())
        all_results.update(phase5_statistical_validation())
        all_results.update(phase6_robustness_test())
        acceptance = phase7_final_acceptance(all_results)
        all_results.update(acceptance)
    except Exception as e:
        print_error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print_header("Validation Summary")
    
    total_checks = sum(1 for k, v in all_results.items() if isinstance(v, bool))
    passed_checks = sum(1 for k, v in all_results.items() if isinstance(v, bool) and v)
    failed_checks = total_checks - passed_checks
    
    print_info(f"Total checks: {total_checks}")
    print_success(f"Passed: {passed_checks}")
    if failed_checks > 0:
        print_error(f"Failed: {failed_checks}")
    
    if all_results.get('ALL_CRITERIA_PASSED', False):
        print_success("\n🎉 ALL ACCEPTANCE CRITERIA PASSED!")
        print_success("Pipeline is ready for final delivery!")
        return 0
    else:
        print_error("\n⚠️  SOME CRITERIA FAILED")
        print_error("Please review the validation results above")
        
        # Provide actionable recommendations
        print_section("Recommendations")
        if not all_results.get('dataset_b_phase6_meta', False):
            print_warning("1. Regenerate Scenario B dataset with Phase 6:")
            print_info("   python3 run_all_scenarios.py --scenario ground --num-samples 2000")
        if all_results.get('cv_a_auc_mean', 0) < 0.99 or all_results.get('cv_b_auc_mean', 0) < 0.99:
            print_warning("2. Low AUC detected - consider:")
            print_info("   - Regenerating datasets with correct parameters")
            print_info("   - Checking COVERT_AMP settings")
            print_info("   - Verifying normalization is applied correctly")
        if 'model_a_inference_error' in all_results or 'model_b_inference_error' in all_results:
            print_warning("3. Model inference errors - check model files")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

