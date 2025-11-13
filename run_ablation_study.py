#!/usr/bin/env python3
"""
🔬 ABLATION STUDY SCRIPT
========================
Systematic ablation study to evaluate the contribution of each component:
- Equalization (Scenario A vs B)
- Attention mechanism (with/without)

Usage:
    python3 run_ablation_study.py
"""

import os
import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')

# Configuration
from config.settings import (
    init_directories,
    DATASET_DIR,
    MODEL_DIR,
    RESULT_DIR,
    GLOBAL_SEED,
    USE_FOCAL_LOSS,
    FOCAL_LOSS_GAMMA,
    FOCAL_LOSS_ALPHA
)

# Set global seeds
from utils.reproducibility import set_global_seeds
set_global_seeds(deterministic=True)

# CNN Detector
from model.detector_cnn import CNNDetector


def load_dataset(scenario='sat'):
    """
    Load dataset for given scenario.
    
    Args:
        scenario: 'sat' (Scenario A) or 'ground' (Scenario B)
    
    Returns:
        X: OFDM grids (N, symbols, subcarriers, 2) or (N, symbols, subcarriers)
        y: Labels (N,)
    """
    # Find dataset file
    dataset_pattern = f"dataset_scenario_{'a' if scenario == 'sat' else 'b'}_*.pkl"
    dataset_files = [f for f in os.listdir(DATASET_DIR) if f.startswith(f"dataset_scenario_{'a' if scenario == 'sat' else 'b'}_") and f.endswith('.pkl')]
    
    if not dataset_files:
        raise FileNotFoundError(f"No dataset found for scenario {scenario}")
    
    # Use the largest dataset (10K if available)
    dataset_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0, reverse=True)
    dataset_path = os.path.join(DATASET_DIR, dataset_files[0])
    
    print(f"  📂 Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract grids and labels
    rx_grids = data['rx_grids']
    labels = data['labels']
    
    # Convert to numpy
    if isinstance(rx_grids, list):
        rx_grids = np.array(rx_grids)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Keep complex data as-is (CNNDetector will handle preprocessing)
    # Ensure proper shape: (N, symbols, subcarriers) - complex
    if rx_grids.ndim == 2:
        rx_grids = rx_grids[np.newaxis, :, :]
    
    print(f"  ✓ Loaded {len(rx_grids)} samples")
    print(f"  ✓ Shape: {rx_grids.shape}")
    print(f"  ✓ Labels: {np.sum(labels==0)} benign, {np.sum(labels==1)} attack")
    
    return rx_grids, labels


def train_and_evaluate(scenario='sat', use_attention=True, epochs=30, batch_size=512):
    """
    Train CNN model and evaluate on test set.
    
    Args:
        scenario: 'sat' (Scenario A) or 'ground' (Scenario B)
        use_attention: Whether to use attention mechanism
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        metrics: Dict with AUC, Precision, Recall, F1
    """
    print(f"\n{'='*60}")
    print(f"🔬 Ablation: Scenario={scenario.upper()}, Attention={'ON' if use_attention else 'OFF'}")
    print(f"{'='*60}")
    
    # Load dataset
    X, y = load_dataset(scenario)
    
    # Split: 80/20 (single split for ablation - faster)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train/val (80/20 of train = 64/16 of total)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"  📊 Train: {len(X_train_final)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
    
    # Build model (let CNNDetector handle preprocessing internally)
    print(f"  🏗️  Building CNN model (attention={'ON' if use_attention else 'OFF'})...")
    
    detector = CNNDetector(
        use_csi=False,
        input_shape=None,  # Auto-detect from data
        learning_rate=0.001,
        dropout_rate=0.3,
        random_state=42,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_LOSS_GAMMA,
        focal_alpha=FOCAL_LOSS_ALPHA,
        use_attention=use_attention
    )
    
    # Train (CNNDetector will handle preprocessing internally)
    print(f"  🚀 Training for {epochs} epochs...")
    detector.train(
        X_train_final, y_train_final,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # Suppress output
    )
    
    # Evaluate (use raw test data, detector will preprocess)
    print("  📈 Evaluating on test set...")
    y_pred_proba = detector.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    print(f"  ✅ Results: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return metrics


def main():
    """Run ablation study."""
    print("="*60)
    print("🔬 ABLATION STUDY: Component Contribution Analysis")
    print("="*60)
    
    # Initialize directories
    init_directories()
    
    # Ablation configurations
    configs = [
        {"scenario": "sat", "eq": False, "att": False, "name": "Baseline (No EQ, No Att)"},
        {"scenario": "sat", "eq": False, "att": True, "name": "+ Attention"},
        {"scenario": "ground", "eq": True, "att": False, "name": "+ Equalization"},
        {"scenario": "ground", "eq": True, "att": True, "name": "+ EQ + Attention"},
    ]
    
    results = {}
    
    # Run experiments
    for cfg in configs:
        scenario = cfg['scenario']
        use_attention = cfg['att']
        
        try:
            metrics = train_and_evaluate(
                scenario=scenario,
                use_attention=use_attention,
                epochs=30,  # Reduced for faster ablation
                batch_size=512
            )
            
            results[cfg['name']] = {
                'scenario': scenario,
                'equalization': cfg['eq'],
                'attention': cfg['att'],
                **metrics
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[cfg['name']] = {
                'scenario': scenario,
                'equalization': cfg['eq'],
                'attention': cfg['att'],
                'error': str(e)
            }
    
    # Print summary table
    print("\n" + "="*60)
    print("📊 ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} {'EQ':<5} {'ATT':<5} {'AUC':<8} {'F1':<8}")
    print("-"*60)
    
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<25} {'✓' if res['equalization'] else '✗':<5} {'✓' if res['attention'] else '✗':<5} {'ERROR':<8} {'ERROR':<8}")
        else:
            print(f"{name:<25} {'✓' if res['equalization'] else '✗':<5} {'✓' if res['attention'] else '✗':<5} {res['auc']:.3f}     {res['f1']:.3f}")
    
    # Save results
    output_path = os.path.join(RESULT_DIR, 'ablation_study_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate LaTeX table
    print("\n" + "="*60)
    print("📝 LaTeX Table for Paper")
    print("="*60)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Ablation Study: Component Contribution Analysis}")
    print("\\label{tab:ablation}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{EQ} & \\textbf{ATT} & \\textbf{AUC (A)} & \\textbf{AUC (B)} \\\\")
    print("\\midrule")
    
    # Group by scenario for table
    scenario_a_results = {k: v for k, v in results.items() if v.get('scenario') == 'sat'}
    scenario_b_results = {k: v for k, v in results.items() if v.get('scenario') == 'ground'}
    
    # Map configurations
    config_map = {
        "Baseline (No EQ, No Att)": "Baseline",
        "+ Attention": "+ Attention",
        "+ Equalization": "+ Equalization",
        "+ EQ + Attention": "+ EQ + Attention"
    }
    
    for cfg_name in ["Baseline (No EQ, No Att)", "+ Attention", "+ Equalization", "+ EQ + Attention"]:
        if cfg_name in scenario_a_results:
            res_a = scenario_a_results[cfg_name]
            auc_a = f"{res_a['auc']:.3f}" if 'auc' in res_a else "N/A"
        else:
            auc_a = "-"
        
        if cfg_name in scenario_b_results:
            res_b = scenario_b_results[cfg_name]
            auc_b = f"{res_b['auc']:.3f}" if 'auc' in res_b else "N/A"
        else:
            auc_b = "-"
        
        eq_symbol = "✓" if (cfg_name in scenario_a_results and scenario_a_results[cfg_name].get('equalization')) or \
                          (cfg_name in scenario_b_results and scenario_b_results[cfg_name].get('equalization')) else "✗"
        att_symbol = "✓" if (cfg_name in scenario_a_results and scenario_a_results[cfg_name].get('attention')) or \
                           (cfg_name in scenario_b_results and scenario_b_results[cfg_name].get('attention')) else "✗"
        
        display_name = config_map.get(cfg_name, cfg_name)
        print(f"{display_name} & {eq_symbol} & {att_symbol} & {auc_a} & {auc_b} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")
    
    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()

