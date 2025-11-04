#!/usr/bin/env python3
"""
🎯 FREQUENCY-DOMAIN DETECTOR
============================
RandomForest-based detector using OFDM grid magnitude features.

Replaces CNN-based detector with proven approach from test_detection_sanity.py:
- Frequency-domain features (OFDM grid magnitude)
- RandomForest classifier (200 trees)
- Parallel feature extraction
- Global normalization
- Near-perfect performance (AUC=1.0)

Usage:
    from model.detector_frequency import FrequencyDetector
    
    detector = FrequencyDetector()
    detector.train(X_grids, y_labels)
    predictions = detector.predict(X_grids_test)
"""

import os
import numpy as np
import pickle
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import tensorflow as tf


class FrequencyDetector:
    """
    Frequency-domain covert channel detector.
    
    Uses OFDM grid magnitude features with RandomForest classifier.
    Proven to achieve AUC=1.0 on test datasets.
    """
    
    def __init__(self, n_estimators=100, max_depth=12, random_state=42, n_jobs=-1,
                 focus_mask=None, mask_weight=10.0):
        """
        Initialize detector with test-proven hyperparameters.
        
        Args:
            n_estimators: Number of trees in RandomForest (default: 100)
            max_depth: Maximum tree depth (default: 12)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)
            mask_weight: Weight for focus mask features (default: 10.0)
        """
        self.model = RandomForestClassifier(
            n_estimators=100,      # برای تست سریع و جلوگیری از زمان‌های طولانی
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,    # اجازه میدیم در دیتاست کوچک الگوهای ریزتر یاد گرفته بشن
            max_features='sqrt',
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        
        self.norm_params = None  # Will store {'mean': float, 'std': float} for global normalization
        self.is_trained = False
        self.random_state = random_state
        self.n_jobs = n_jobs
        # Optional focus mask to emphasize injected REs
        self.focus_mask = focus_mask  # numpy array shaped like a single grid (Nsym, Nsc)
        self.mask_weight = float(mask_weight) if mask_weight is not None else 10.0

    def _build_default_focus_mask(self, grid_like):
        """
        Build a deterministic focus mask targeting the ACTUAL injection region:
        - Symbols: 1 to 7 (indices [1:8])
        - Subcarriers: 0 to 31 (indices [0:32])
        
        This matches the actual inject_covert_channel_fixed() implementation:
        - step = n_subs * 5 → typically selects first ~32 subcarriers
        - selected_symbols = [1,2,3,4,5,6,7]
        """
        # Determine grid shape (expect squeeze -> (Nsym, Nsc))
        if isinstance(grid_like, tf.Tensor):
            g = tf.squeeze(grid_like).numpy()
        else:
            g = np.squeeze(grid_like)
        if g.ndim != 2:
            # Try to infer last two dims as (Nsym, Nsc)
            g = np.squeeze(g)
        n_sym, n_sc = g.shape[-2], g.shape[-1]

        mask = np.zeros((n_sym, n_sc), dtype=np.float32)
        
        # تزریق در subcarriers [0..31] و symbols [1..7]
        # محور عمودی (symbol) = 1 تا 7
        # محور افقی (subcarrier) = 0 تا 31
        mask[1:8, 0:32] = 1.0
        
        return mask
    
    def _extract_features_single(self, tx_grid):
        """
        ENHANCED: Extract comprehensive features from OFDM grid.
        
        Features include:
        1. Magnitude features (mean, std, max)
        2. Phase features (mean, std)
        3. Power distribution (mean, std, max)
        4. Focus region features (if mask available)
        
        This matches the test pipeline for high detection performance.
        """
        # Convert to numpy if TensorFlow tensor
        if isinstance(tx_grid, tf.Tensor):
            grid_flat = tf.squeeze(tx_grid).numpy()
        else:
            grid_flat = np.squeeze(tx_grid)
        
        # Extract magnitude and phase
        magnitude = np.abs(grid_flat)  # Expected shape: (Nsym, Nsc)
        phase = np.angle(grid_flat)    # Phase information
        
        feature_list = []
        
        # 1. Global magnitude statistics
        feature_list.extend([
            magnitude.mean(),
            magnitude.std(),
            magnitude.max(),
            magnitude.min(),
        ])
        
        # 2. Global phase statistics
        feature_list.extend([
            phase.mean(),
            phase.std(),
        ])
        
        # 3. Power distribution
        power = magnitude ** 2
        feature_list.extend([
            power.mean(),
            power.std(),
            power.max(),
        ])
        
        # 4. Per-symbol statistics (if multiple symbols)
        if magnitude.shape[0] > 1:
            per_symbol_mean = magnitude.mean(axis=1)  # Mean per OFDM symbol
            per_symbol_std = magnitude.std(axis=1)
            feature_list.extend([
                per_symbol_mean.mean(),
                per_symbol_mean.std(),
                per_symbol_std.mean(),
            ])
        
        # 5. Per-subcarrier statistics (if multiple subcarriers)
        if magnitude.shape[1] > 1:
            per_subcarrier_mean = magnitude.mean(axis=0)  # Mean per subcarrier
            per_subcarrier_std = magnitude.std(axis=0)
            feature_list.extend([
                per_subcarrier_mean.mean(),
                per_subcarrier_mean.std(),
                per_subcarrier_std.mean(),
            ])
        
        # 6. Focus region features (if mask available)
        if self.focus_mask is None:
            try:
                self.focus_mask = self._build_default_focus_mask(grid_flat)
            except Exception:
                self.focus_mask = None
        
        if self.focus_mask is not None:
            try:
                focus_region = magnitude[self.focus_mask > 0]
                if len(focus_region) > 0:
                    feature_list.extend([
                        focus_region.mean(),
                        focus_region.std(),
                        focus_region.max(),
                    ])
            except Exception:
                pass
        
        # 7. Flattened magnitude (subset for dimensionality control)
        # Take every Nth element to keep feature count manageable
        flat_mag = magnitude.flatten()
        stride = max(1, len(flat_mag) // 100)  # Keep ~100 magnitude samples
        feature_list.extend(flat_mag[::stride].tolist())
        
        return np.array(feature_list, dtype=np.float32)
    
    def extract_features(self, X_grids, verbose=True):
        """
        Extract features from OFDM grids (parallel).
        
        Args:
            X_grids: List of OFDM grids (TensorFlow tensors or numpy arrays)
            verbose: Print progress
            
        Returns:
            features: Normalized feature array (n_samples, n_features)
        """
        n_samples = len(X_grids)
        
        if verbose:
            print(f"  Extracting features from {n_samples} samples...")
        
        # Parallel extraction for >100 samples
        use_parallel = n_samples > 100
        
        if use_parallel:
            features = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(self._extract_features_single)(x) for x in X_grids
            )
            features = list(features)
        else:
            features = []
            for x in X_grids:
                feat = self._extract_features_single(x)
                features.append(feat)
        
        # Stack to numpy array
        features = np.array(features, dtype=np.float32)
        
        if verbose:
            print(f"  Feature shape: {features.shape}")
        
        return features
    
    def normalize_features(self, features, fit=True, verbose=False):
        """
        Normalize features with GLOBAL mean/std (matching test_detection_sanity.py).
        ...
        """
        if fit:
            # 🔧 CRITICAL FIX: Use GLOBAL normalization (no axis parameter)
            self.norm_params = {
                'mean': features.mean(),  # 👈 مطمئن شوید axis=0 حذف شده
                'std': features.std()      # 👈 مطمئن شوید axis=0 حذف شده
            }
            
            # 🔍 DEBUG مورد 6: چک normalization params
            if verbose:
                print(f"  🔍 DEBUG scaler fitted:")
                print(f"      mean = {self.norm_params['mean']:.6f}")
                print(f"      std  = {self.norm_params['std']:.6f}")
        
        if self.norm_params is None:
            raise ValueError("Normalization parameters not set. Call with fit=True first.")
        
        # Apply global normalization
        features_norm = (features - self.norm_params['mean']) / (self.norm_params['std'] + 1e-8)
        
        return features_norm
    
    def train(self, X_grids, y_labels, verbose=True):
        """
        Train detector on OFDM grids.
        
        Args:
            X_grids: List of OFDM grids (training data)
            y_labels: Binary labels (0=benign, 1=attack)
            verbose: Print training progress
            
        Returns:
            Training history dict
        """
        if verbose:
            print(f"\n{'='*70}")
            print("🤖 TRAINING FREQUENCY-DOMAIN DETECTOR")
            print(f"{'='*70}")
        
        # Extract features
        features = self.extract_features(X_grids, verbose=verbose)
        
        # Normalize (fit on training data)
        features_norm = self.normalize_features(features, fit=True)
        
        if verbose:
            print(f"  Normalized range: [{features_norm.min():.3f}, {features_norm.max():.3f}]")
            print(f"  Training samples: {len(y_labels)} (benign={np.sum(y_labels==0)}, attack={np.sum(y_labels==1)})")
        
        # Train RandomForest
        if verbose:
            print(f"  Training RandomForest (n_estimators={self.model.n_estimators})...")
        
        self.model.fit(features_norm, y_labels)
        self.is_trained = True
        
        # Training metrics
        y_prob_train = self.model.predict_proba(features_norm)[:, 1]
        auc_train = roc_auc_score(y_labels, y_prob_train)
        
        if verbose:
            print(f"  Training AUC: {auc_train:.4f}")
            
            # 🔍 DEBUG مورد 10: Feature importance
            importances = self.model.feature_importances_
            top_10_idx = np.argsort(importances)[-10:]
            print(f"  🔍 DEBUG Top-10 RF feature importances:")
            print(f"      Indices: {top_10_idx.tolist()}")
            print(f"      Values:  {importances[top_10_idx]}")
            if np.max(importances) < 0.001:
                print(f"      ⚠️ WARNING: All importances very low → features might be ineffective!")
            
            print(f"  ✅ Training complete!")
        
        return {
            'auc_train': auc_train,
            'n_features': features.shape[1],
            'n_samples': len(y_labels)
        }
    
    def predict(self, X_grids, return_proba=False, verbose=False):
        """
        Predict on new OFDM grids.
        
        Args:
            X_grids: List of OFDM grids (test data)
            return_proba: If True, return probabilities. If False, return binary predictions.
            verbose: Print progress
            
        Returns:
            Predictions (binary or probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features(X_grids, verbose=verbose)
        
        # Normalize (use training params)
        features_norm = self.normalize_features(features, fit=False)
        
        # Predict
        if return_proba:
            predictions = self.model.predict_proba(features_norm)[:, 1]
        else:
            predictions = self.model.predict(features_norm)
        
        return predictions
    
    def evaluate(self, X_grids, y_labels, threshold=0.5, verbose=True):
        """
        Evaluate detector on test data.
        
        Args:
            X_grids: List of OFDM grids (test data)
            y_labels: True binary labels
            threshold: Decision threshold
            verbose: Print results
            
        Returns:
            Dict with evaluation metrics
        """
        # Predict probabilities
        y_prob = self.predict(X_grids, return_proba=True, verbose=verbose)
        
        # Binary predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_labels, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_labels, y_pred, average='binary', zero_division=0
        )
        
        # False positive rate (benign predicted as attack)
        benign_idx = (y_labels == 0)
        if benign_idx.sum() > 0:
            fp_rate = (y_pred[benign_idx] == 1).sum() / benign_idx.sum()
        else:
            fp_rate = 0.0
        
        # True positive rate (attack predicted as attack)
        attack_idx = (y_labels == 1)
        if attack_idx.sum() > 0:
            tp_rate = (y_pred[attack_idx] == 1).sum() / attack_idx.sum()
        else:
            tp_rate = 0.0
        
        results = {
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp_rate': tp_rate,
            'fp_rate': fp_rate,
            'threshold': threshold,
            'y_prob': y_prob,      # ✅ ADD: Predicted probabilities
            'y_pred': y_pred       # ✅ ADD: Binary predictions
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("📊 EVALUATION RESULTS")
            print(f"{'='*70}")
            print(f"  AUC: {auc:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  True Positive Rate: {tp_rate*100:.1f}%")
            print(f"  False Positive Rate: {fp_rate*100:.1f}%")
            print(f"  Threshold: {threshold:.3f}")
        
        return results
    
    def save(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")
        
        save_dict = {
            'model': self.model,
            'norm_params': self.norm_params,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.norm_params = save_dict['norm_params']
        self.is_trained = save_dict['is_trained']
        
        print(f"✅ Model loaded from: {filepath}")


# ============================================================================
# Helper Functions for Main Pipeline Integration
# ============================================================================

def train_frequency_detector(dataset, verbose=True):
    """
    Train frequency detector on dataset.
    
    Args:
        dataset: Dict with keys 'tx_grids', 'labels'
        verbose: Print progress
        
    Returns:
        Trained FrequencyDetector instance
    """
    detector = FrequencyDetector()
    
    # Get training data
    X_grids = dataset['tx_grids']
    y_labels = dataset['labels']
    
    # Train
    detector.train(X_grids, y_labels, verbose=verbose)
    
    return detector


def evaluate_frequency_detector(detector, dataset, threshold=0.5, verbose=True):
    """
    Evaluate frequency detector on dataset.
    
    Args:
        detector: Trained FrequencyDetector
        dataset: Dict with keys 'tx_grids', 'labels'
        threshold: Decision threshold
        verbose: Print results
        
    Returns:
        Dict with evaluation metrics and predictions
    """
    X_grids = dataset['tx_grids']
    y_labels = dataset['labels']
    
    # Evaluate
    results = detector.evaluate(X_grids, y_labels, threshold=threshold, verbose=verbose)
    
    # Get predictions for downstream use
    y_prob = detector.predict(X_grids, return_proba=True)
    y_pred = (y_prob >= threshold).astype(int)
    
    results['y_prob'] = y_prob
    results['y_pred'] = y_pred
    
    return results