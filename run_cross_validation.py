#!/usr/bin/env python3
"""
5-Fold Stratified Cross-Validation for CNN Detector
====================================================
Provides robust evaluation with confidence intervals
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import tensorflow as tf

# Import from main_detection_cnn
import sys
sys.path.insert(0, str(Path.cwd()))


def load_dataset(scenario='sat'):
    """Load dataset for cross-validation."""
    import glob
    import re
    
    scenario_letter = 'a' if scenario == 'sat' else 'b'
    pattern = f'dataset/dataset_scenario_{scenario_letter}_*.pkl'
    candidates = glob.glob(pattern)
    
    if not candidates:
        raise FileNotFoundError(f"No dataset found for scenario {scenario}")
    
    # Sort by sample count (descending)
    def extract_samples(path):
        match = re.search(r'dataset_scenario_[ab]_(\d+)\.pkl', path)
        return int(match.group(1)) if match else 0
    
    candidates.sort(key=extract_samples, reverse=True)
    dataset_file = Path(candidates[0])
    
    print(f"  Loading: {dataset_file.name}")
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset


def build_cnn_model(input_shape):
    """Build CNN model architecture (same as main_detection_cnn.py)."""
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_and_evaluate_fold(X_train, y_train, X_val, y_val, fold_num, scenario):
    """Train and evaluate one fold."""
    print(f"\n    {'='*70}")
    print(f"    Fold {fold_num}/5")
    print(f"    {'='*70}")
    
    # Handle complex data
    if np.iscomplexobj(X_train):
        print(f"      Converting complex to magnitude...")
        X_train = np.abs(X_train)
        X_val = np.abs(X_val)
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    X_train_norm = scaler.fit_transform(X_train_flat)
    X_val_norm = scaler.transform(X_val_flat)
    
    # Reshape back
    X_train_norm = X_train_norm.reshape(X_train.shape)
    X_val_norm = X_val_norm.reshape(X_val.shape)
    
    # Add channel dimension if needed
    if X_train_norm.ndim == 3:
        X_train_norm = np.expand_dims(X_train_norm, axis=-1)
        X_val_norm = np.expand_dims(X_val_norm, axis=-1)
    
    # Build model
    input_shape = X_train_norm.shape[1:]
    model = build_cnn_model(input_shape)
    
    print(f"      Training samples: {len(X_train)}")
    print(f"      Validation samples: {len(X_val)}")
    print(f"      Input shape: {input_shape}")
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train_norm, y_train,
        validation_data=(X_val_norm, y_val),
        epochs=100,
        batch_size=512,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_val_norm, verbose=0).flatten()
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    
    auc = roc_auc_score(y_val, y_pred_prob)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val == 0))
    fn = np.sum((y_pred == 0) & (y_val == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"      ✓ AUC: {auc:.4f}")
    print(f"      ✓ Precision: {precision:.4f}")
    print(f"      ✓ Recall: {recall:.4f}")
    print(f"      ✓ F1: {f1:.4f}")
    print(f"      ✓ Epochs trained: {len(history.history['loss'])}")
    
    return {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(optimal_threshold),
        'epochs': len(history.history['loss']),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val))
    }


def run_cross_validation(scenario='sat', n_folds=5):
    """Run stratified k-fold cross-validation."""
    print(f"\n{'='*80}")
    print(f"🔄 5-Fold Stratified Cross-Validation")
    print(f"{'='*80}")
    print(f"Scenario: {scenario.upper()}")
    print(f"Folds: {n_folds}")
    print()
    
    # Load dataset
    dataset = load_dataset(scenario)
    X = dataset['rx_grids']
    y = np.array(dataset['labels'])
    
    print(f"  Dataset: {len(y)} samples")
    print(f"  Shape: {X.shape}")
    print(f"  Benign: {np.sum(y == 0)}, Attack: {np.sum(y == 1)}")
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results
    fold_results = []
    
    # Run each fold
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        result = train_and_evaluate_fold(
            X_train, y_train, X_val, y_val,
            fold_num, scenario
        )
        
        fold_results.append(result)
    
    # Aggregate results
    metrics = ['auc', 'precision', 'recall', 'f1']
    aggregated = {}
    
    print(f"\n{'='*80}")
    print("📊 Cross-Validation Results")
    print(f"{'='*80}")
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        aggregated[metric] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'values': [float(v) for v in values]
        }
        
        print(f"  {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"{'='*80}")
    
    return {
        'scenario': scenario,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'aggregated': aggregated,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Run cross-validation for both scenarios."""
    print("="*80)
    print("🎯 5-Fold Stratified Cross-Validation for CNN Detector")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Scenario A
    try:
        print("\n" + "🔵"*40)
        print("SCENARIO A: Single-hop Downlink (Insider@Satellite)")
        print("🔵"*40)
        results['scenario_a'] = run_cross_validation('sat', n_folds=5)
    except Exception as e:
        print(f"❌ Scenario A failed: {e}")
        results['scenario_a'] = {'error': str(e)}
    
    # Scenario B
    try:
        print("\n" + "🟢"*40)
        print("SCENARIO B: Two-hop Relay (Insider@Ground)")
        print("🟢"*40)
        results['scenario_b'] = run_cross_validation('ground', n_folds=5)
    except Exception as e:
        print(f"❌ Scenario B failed: {e}")
        results['scenario_b'] = {'error': str(e)}
    
    # Save results
    output_dir = Path('result')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'cross_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ Cross-Validation Completed")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Summary
    print("\n📊 FINAL SUMMARY")
    print("="*80)
    
    for scenario_key, scenario_name in [('scenario_a', 'Scenario A'), ('scenario_b', 'Scenario B')]:
        if scenario_key in results and 'aggregated' in results[scenario_key]:
            agg = results[scenario_key]['aggregated']
            print(f"\n{scenario_name}:")
            print(f"  AUC:       {agg['auc']['mean']:.4f} ± {agg['auc']['std']:.4f}")
            print(f"  Precision: {agg['precision']['mean']:.4f} ± {agg['precision']['std']:.4f}")
            print(f"  Recall:    {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")
            print(f"  F1 Score:  {agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

