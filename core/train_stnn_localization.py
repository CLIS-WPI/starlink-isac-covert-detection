# ======================================
# 📄 core/train_stnn_localization.py
# Purpose: Training pipeline for STNN models (TDOA/FDOA)
# Multi-GPU: Uses MirroredStrategy to train on both H100 GPUs
# Validation: Anderson-Darling test for error distribution
# ======================================

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Tuple, Dict
import matplotlib.pyplot as plt

# Adjust path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.stnn_localization import (
    build_stnn_tdoa_model, 
    build_stnn_fdoa_model,
    create_training_callbacks,
    STNNEstimator
)
from utils.stft_features import extract_stft_features_from_dataset
from config.settings import MODEL_DIR, DATASET_DIR, NUM_SAMPLES_PER_CLASS, NUM_SATELLITES_FOR_TDOA


def anderson_darling_test(errors: np.ndarray, 
                           alpha: float = 0.05) -> Tuple[bool, float, str]:
    """
    Perform Anderson-Darling test to check if errors follow normal distribution.
    
    As per paper: "We validate the normal distribution assumption of the 
    estimation errors using Anderson-Darling test."
    
    Args:
        errors: Array of estimation errors (true - predicted)
        alpha: Significance level (0.05 = 95% confidence)
    
    Returns:
        (is_normal, statistic, interpretation)
        - is_normal: Whether null hypothesis (normal dist) is accepted
        - statistic: Test statistic
        - interpretation: Human-readable result
    """
    # Normalize errors
    errors_normalized = (errors - np.mean(errors)) / (np.std(errors) + 1e-12)
    
    # Anderson-Darling test
    result = stats.anderson(errors_normalized, dist='norm')
    
    # Check against critical value for given alpha
    # Anderson returns critical values for [15%, 10%, 5%, 2.5%, 1%]
    alpha_to_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
    idx = alpha_to_idx.get(alpha, 2)  # Default to 5%
    
    critical_value = result.critical_values[idx]
    is_normal = result.statistic < critical_value
    
    interpretation = (
        f"Anderson-Darling statistic: {result.statistic:.4f}\n"
        f"Critical value (α={alpha}): {critical_value:.4f}\n"
        f"Result: {'✓ PASS' if is_normal else '✗ FAIL'} - "
        f"Errors {'follow' if is_normal else 'do NOT follow'} normal distribution "
        f"at {(1-alpha)*100}% confidence"
    )
    
    return is_normal, result.statistic, interpretation


def plot_error_distribution(errors: np.ndarray, 
                              title: str = "Error Distribution",
                              save_path: str = None):
    """
    Plot error distribution with Q-Q plot (as per paper's Figure 3).
    
    Args:
        errors: Array of errors
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability density
    ax1.hist(errors, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2, label='Normal Fit')
    
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'{title} - Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Error distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def prepare_stnn_data(dataset_path: str, 
                       test_size: float = 0.2,
                       val_size: float = 0.1) -> Dict:
    """
    Prepare training/validation/test data for STNN.
    
    Args:
        dataset_path: Path to pickled dataset
        test_size: Fraction for test set
        val_size: Fraction of training set for validation
    
    Returns:
        Dictionary with train/val/test splits and normalization params
    """
    print("\n" + "="*60)
    print("STNN DATA PREPARATION")
    print("="*60)
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"✓ Dataset loaded: {len(dataset['labels'])} samples")
    
    # Extract STFT features and labels
    print(f"\n[2/4] Extracting STFT features...")
    stft_features, tdoa_labels, fdoa_labels = extract_stft_features_from_dataset(
        dataset, 
        fs=dataset.get('sampling_rate', 38400),
        output_shape=(256, 256)
    )
    
    # Normalization parameters (paper's scheme from Table 3)
    tdoa_max = np.max(np.abs(tdoa_labels))
    fdoa_max = np.max(np.abs(fdoa_labels))
    
    tdoa_labels_norm = tdoa_labels / tdoa_max
    fdoa_labels_norm = fdoa_labels / fdoa_max
    
    print(f"\n[3/4] Normalization:")
    print(f"  TDOA max: {tdoa_max*1e6:.2f} μs")
    print(f"  FDOA max: {fdoa_max:.2f} Hz")
    print(f"  TDOA normalized range: [{tdoa_labels_norm.min():.4f}, {tdoa_labels_norm.max():.4f}]")
    print(f"  FDOA normalized range: [{fdoa_labels_norm.min():.4f}, {fdoa_labels_norm.max():.4f}]")
    
    # Train/test split
    print(f"\n[4/4] Splitting data...")
    X_trainval, X_test, y_tdoa_trainval, y_tdoa_test, y_fdoa_trainval, y_fdoa_test = train_test_split(
        stft_features, tdoa_labels_norm, fdoa_labels_norm,
        test_size=test_size, random_state=42, shuffle=True
    )
    
    # Train/val split
    X_train, X_val, y_tdoa_train, y_tdoa_val, y_fdoa_train, y_fdoa_val = train_test_split(
        X_trainval, y_tdoa_trainval, y_fdoa_trainval,
        test_size=val_size, random_state=42, shuffle=True
    )
    
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Val:   {len(X_val)} samples")
    print(f"✓ Test:  {len(X_test)} samples")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_tdoa_train': y_tdoa_train, 'y_tdoa_val': y_tdoa_val, 'y_tdoa_test': y_tdoa_test,
        'y_fdoa_train': y_fdoa_train, 'y_fdoa_val': y_fdoa_val, 'y_fdoa_test': y_fdoa_test,
        'tdoa_max': tdoa_max, 'fdoa_max': fdoa_max,
        'tdoa_labels_raw': tdoa_labels, 'fdoa_labels_raw': fdoa_labels  # For denormalization
    }


def train_stnn_tdoa(data: Dict, 
                     epochs: int = 50,
                     batch_size: int = 32,
                     use_multi_gpu: bool = True) -> Tuple[tf.keras.Model, Dict]:
    """
    Train STNN model for TDOA estimation.
    
    Uses MirroredStrategy for multi-GPU training on both H100s.
    
    Args:
        data: Data dictionary from prepare_stnn_data()
        epochs: Training epochs
        batch_size: Batch size (per GPU)
        use_multi_gpu: Use both GPUs if True
    
    Returns:
        (trained_model, history_dict)
    """
    print("\n" + "="*60)
    print("TRAINING STNN FOR TDOA")
    print("="*60)
    
    # Multi-GPU strategy
    if use_multi_gpu:
        try:
            strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
            print(f"\n✓ Multi-GPU enabled: {strategy.num_replicas_in_sync} GPUs")
        except:
            print("\n⚠️  Multi-GPU failed, using single GPU")
            strategy = tf.distribute.get_strategy()  # Default strategy
    else:
        strategy = tf.distribute.get_strategy()
    
    # Build model within strategy scope
    with strategy.scope():
        model = build_stnn_tdoa_model(input_shape=(256, 256, 1))
        print(f"\n✓ Model built with {model.count_params():,} parameters")
    
    # Callbacks
    callbacks_list = create_training_callbacks('tdoa', MODEL_DIR)
    
    # Training
    print(f"\n[Training] Epochs: {epochs}, Batch size: {batch_size}")
    history = model.fit(
        data['X_train'], data['y_tdoa_train'],
        validation_data=(data['X_val'], data['y_tdoa_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, 'stnn_tdoa_final.keras')
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    return model, history.history


def train_stnn_fdoa(data: Dict, 
                     epochs: int = 50,
                     batch_size: int = 32,
                     use_multi_gpu: bool = True) -> Tuple[tf.keras.Model, Dict]:
    """
    Train STNN model for FDOA estimation.
    
    Uses MirroredStrategy for multi-GPU training on both H100s.
    
    Args:
        data: Data dictionary from prepare_stnn_data()
        epochs: Training epochs
        batch_size: Batch size (per GPU)
        use_multi_gpu: Use both GPUs if True
    
    Returns:
        (trained_model, history_dict)
    """
    print("\n" + "="*60)
    print("TRAINING STNN FOR FDOA")
    print("="*60)
    
    # Multi-GPU strategy
    if use_multi_gpu:
        try:
            strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
            print(f"\n✓ Multi-GPU enabled: {strategy.num_replicas_in_sync} GPUs")
        except:
            print("\n⚠️  Multi-GPU failed, using single GPU")
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()
    
    # Build model within strategy scope
    with strategy.scope():
        model = build_stnn_fdoa_model(input_shape=(256, 256, 1))
        print(f"\n✓ Model built with {model.count_params():,} parameters")
    
    # Callbacks
    callbacks_list = create_training_callbacks('fdoa', MODEL_DIR)
    
    # Training
    print(f"\n[Training] Epochs: {epochs}, Batch size: {batch_size}")
    history = model.fit(
        data['X_train'], data['y_fdoa_train'],
        validation_data=(data['X_val'], data['y_fdoa_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, 'stnn_fdoa_final.keras')
    model.save(final_path)
    print(f"\n✓ Final model saved to {final_path}")
    
    return model, history.history


def validate_stnn_errors(model: tf.keras.Model,
                          X_val: np.ndarray,
                          y_val_norm: np.ndarray,
                          y_val_raw: np.ndarray,
                          param_max: float,
                          param_name: str = "TDOA",
                          save_dir: str = MODEL_DIR) -> Tuple[float, bool]:
    """
    Validate STNN error distribution using Anderson-Darling test.
    
    Args:
        model: Trained STNN model
        X_val: Validation features
        y_val_norm: Validation labels (normalized)
        y_val_raw: Validation labels (raw, for computing actual errors)
        param_max: Maximum parameter value for denormalization
        param_name: Parameter name (TDOA/FDOA)
        save_dir: Directory to save plots
    
    Returns:
        (error_std, is_normal)
        - error_std: Standard deviation of errors
        - is_normal: Whether errors follow normal distribution
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING {param_name} ERROR DISTRIBUTION")
    print(f"{'='*60}")
    
    # Predict
    y_pred_norm = model.predict(X_val, verbose=0).flatten()
    y_pred = y_pred_norm * param_max  # Denormalize
    
    # Compute errors (true - predicted)
    errors = y_val_raw - y_pred
    
    # Statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mae = np.mean(np.abs(errors))
    
    print(f"\n[Error Statistics]")
    if param_name == "TDOA":
        print(f"  Mean error: {mean_error*1e6:.4f} μs")
        print(f"  Std error: {std_error*1e6:.4f} μs")
        print(f"  MAE: {mae*1e6:.4f} μs")
        print(f"  ±3σ range: {3*std_error*1e6:.4f} μs")
    else:  # FDOA
        print(f"  Mean error: {mean_error:.4f} Hz")
        print(f"  Std error: {std_error:.4f} Hz")
        print(f"  MAE: {mae:.4f} Hz")
        print(f"  ±3σ range: {3*std_error:.4f} Hz")
    
    # Anderson-Darling test
    print(f"\n[Anderson-Darling Test]")
    is_normal, statistic, interpretation = anderson_darling_test(errors, alpha=0.05)
    print(interpretation)
    
    # Plot error distribution
    plot_save_path = os.path.join(save_dir, f'stnn_{param_name.lower()}_error_dist.png')
    plot_error_distribution(errors, title=f"{param_name} Errors", save_path=plot_save_path)
    
    # ✅ NEW: Save error stats for CAF refinement (Section 3.2 ICCIP 2024)
    save_stnn_error_stats_for_caf(
        y_true=y_val_raw,
        y_pred=y_pred,
        param_name=param_name,
        save_dir=save_dir
    )
    
    return std_error, is_normal


def save_stnn_error_stats_for_caf(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    param_name: str,
                                    save_dir: str):
    """
    Compute and save mean and std of STNN estimation errors for CAF refinement.
    Based on Section 3.2 of ICCIP 2024.
    
    Args:
        y_true: Ground truth values
        y_pred: STNN predictions
        param_name: "TDOA" or "FDOA"
        save_dir: Directory to save stats
    """
    errors = y_true - y_pred
    mu_e = np.mean(errors)
    sigma_e = np.std(errors)
    
    # Save stats for CAF refinement
    stats_path = os.path.join(save_dir, f'stnn_error_stats_{param_name.lower()}.npz')
    np.savez(stats_path, mu_e=mu_e, sigma_e=sigma_e)
    
    if param_name == "TDOA":
        print(f"[STNN CAF] Error stats saved: μ={mu_e*1e6:.4f} μs, σ={sigma_e*1e6:.4f} μs → {stats_path}")
    else:
        print(f"[STNN CAF] Error stats saved: μ={mu_e:.4f} Hz, σ={sigma_e:.4f} Hz → {stats_path}")


def main_training_pipeline(dataset_path: str = None,
                             epochs_tdoa: int = 50,
                             epochs_fdoa: int = 50,
                             batch_size: int = 32,
                             use_multi_gpu: bool = True):
    """
    Complete training pipeline for STNN models.
    
    Args:
        dataset_path: Path to dataset pickle
        epochs_tdoa: Training epochs for TDOA model
        epochs_fdoa: Training epochs for FDOA model
        batch_size: Batch size
        use_multi_gpu: Use both H100 GPUs
    """
    print("\n" + "="*60)
    print("STNN TRAINING PIPELINE")
    print("="*60)
    
    # Default dataset path
    if dataset_path is None:
        dataset_path = (
            f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_"
            f"sats{NUM_SATELLITES_FOR_TDOA}.pkl"
        )
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please run generate_dataset_parallel.py first!")
        return
    
    # Prepare data
    data = prepare_stnn_data(dataset_path)
    
    # Train TDOA model
    tdoa_model, tdoa_history = train_stnn_tdoa(
        data, epochs=epochs_tdoa, batch_size=batch_size, use_multi_gpu=use_multi_gpu
    )
    
    # Validate TDOA errors
    # Get raw validation labels
    y_tdoa_val_raw = data['y_tdoa_val'] * data['tdoa_max']
    tdoa_std, tdoa_is_normal = validate_stnn_errors(
        tdoa_model, data['X_val'], data['y_tdoa_val'], y_tdoa_val_raw,
        data['tdoa_max'], param_name="TDOA", save_dir=MODEL_DIR
    )
    
    # Train FDOA model
    fdoa_model, fdoa_history = train_stnn_fdoa(
        data, epochs=epochs_fdoa, batch_size=batch_size, use_multi_gpu=use_multi_gpu
    )
    
    # Validate FDOA errors
    y_fdoa_val_raw = data['y_fdoa_val'] * data['fdoa_max']
    fdoa_std, fdoa_is_normal = validate_stnn_errors(
        fdoa_model, data['X_val'], data['y_fdoa_val'], y_fdoa_val_raw,
        data['fdoa_max'], param_name="FDOA", save_dir=MODEL_DIR
    )
    
    # Save error statistics
    stats_path = os.path.join(MODEL_DIR, 'stnn_error_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'tdoa_std': tdoa_std,
            'fdoa_std': fdoa_std,
            'tdoa_is_normal': tdoa_is_normal,
            'fdoa_is_normal': fdoa_is_normal,
            'tdoa_max': data['tdoa_max'],
            'fdoa_max': data['fdoa_max']
        }, f)
    print(f"\n✓ Error statistics saved to {stats_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTDOA Model:")
    print(f"  ✓ Best val loss: {min(tdoa_history['val_loss']):.6f}")
    print(f"  ✓ Error std: {tdoa_std*1e6:.4f} μs")
    print(f"  ✓ Normal distribution: {'YES ✓' if tdoa_is_normal else 'NO ✗'}")
    
    print(f"\nFDOA Model:")
    print(f"  ✓ Best val loss: {min(fdoa_history['val_loss']):.6f}")
    print(f"  ✓ Error std: {fdoa_std:.4f} Hz")
    print(f"  ✓ Normal distribution: {'YES ✓' if fdoa_is_normal else 'NO ✗'}")
    
    print(f"\n✓ Models saved to {MODEL_DIR}/")
    print(f"✓ Ready for integration with localization pipeline!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train STNN models for TDOA/FDOA estimation')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset pickle')
    parser.add_argument('--epochs-tdoa', type=int, default=50, help='Epochs for TDOA training')
    parser.add_argument('--epochs-fdoa', type=int, default=50, help='Epochs for FDOA training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--single-gpu', action='store_true', help='Use single GPU instead of both')
    
    args = parser.parse_args()
    
    main_training_pipeline(
        dataset_path=args.dataset,
        epochs_tdoa=args.epochs_tdoa,
        epochs_fdoa=args.epochs_fdoa,
        batch_size=args.batch_size,
        use_multi_gpu=not args.single_gpu
    )