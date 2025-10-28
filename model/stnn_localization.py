# ======================================
# 📄 model/stnn_localization.py
# Purpose: STNN (STFT + ResNet CNN) for fast TDOA/FDOA estimation
# Based on: "A High-efficiency TDOA and FDOA Estimation Method Based on CNNs" (ICCIP 2024)
# Multi-GPU: Uses MirroredStrategy for training on both H100s
# ======================================

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
from typing import Tuple, Optional


class ResidualBlock(layers.Layer):
    """
    Residual block for STNN (from paper's architecture)
    """
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, 
                                    padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, strides=1, 
                                    padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut projection if dimensions change
        self.shortcut = None
        if strides != 1:
            self.shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')
            self.shortcut_bn = layers.BatchNormalization()
        
        self.activation = layers.Activation('relu')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        # Ensure shapes match
        if x.shape[1:] != shortcut.shape[1:]:
            # Adaptive pooling or padding if needed
            if x.shape[1] < shortcut.shape[1]:
                shortcut = layers.AveragePooling2D(pool_size=2)(shortcut)
            else:
                # Match channels
                if x.shape[-1] != shortcut.shape[-1]:
                    shortcut = layers.Conv2D(x.shape[-1], 1)(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
        })
        return config


def build_stnn_tdoa_model(input_shape=(256, 256, 1), 
                          name='stnn_tdoa') -> Model:
    """
    Build STNN model for TDOA estimation.
    
    Architecture (from paper):
    - STFT input (256x256)
    - 4 Residual blocks with increasing filters
    - Global Average Pooling
    - 2 FC layers
    - Output: Normalized TDOA
    
    Args:
        input_shape: Input STFT shape (H, W, C)
        name: Model name
    
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name=f'{name}_input')
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # 4 Residual blocks (like paper's Figure 2)
    for filters in [64, 128, 256, 512]:
        x = ResidualBlock(filters, strides=2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output: Normalized TDOA (-1 to 1 range)
    tdoa_output = layers.Dense(1, activation='tanh', name='tdoa_output')(x)
    
    model = Model(inputs, tdoa_output, name=name)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def build_stnn_fdoa_model(input_shape=(256, 256, 1), 
                          name='stnn_fdoa') -> Model:
    """
    Build STNN model for FDOA estimation.
    
    Same architecture as TDOA model but trained separately
    (as per paper's approach).
    
    Args:
        input_shape: Input STFT shape (H, W, C)
        name: Model name
    
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name=f'{name}_input')
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # 4 Residual blocks
    for filters in [64, 128, 256, 512]:
        x = ResidualBlock(filters, strides=2)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output: Normalized FDOA (-1 to 1 range)
    fdoa_output = layers.Dense(1, activation='tanh', name='fdoa_output')(x)
    
    model = Model(inputs, fdoa_output, name=name)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


class STNNEstimator:
    """
    Wrapper class for TDOA/FDOA estimation using trained STNN models.
    
    Includes normalization/denormalization and error statistics tracking.
    """
    def __init__(self, 
                 tdoa_model_path: Optional[str] = None,
                 fdoa_model_path: Optional[str] = None,
                 tdoa_max: float = 60 * (1/38400),  # 60 * Ts (from paper)
                 fdoa_max: float = 6144.0):  # Hz (from paper)
        """
        Initialize STNN estimator.
        
        Args:
            tdoa_model_path: Path to trained TDOA model
            fdoa_model_path: Path to trained FDOA model
            tdoa_max: Maximum TDOA value for normalization (seconds)
            fdoa_max: Maximum FDOA value for normalization (Hz)
        """
        self.tdoa_max = tdoa_max
        self.fdoa_max = fdoa_max
        
        # Load models
        self.tdoa_model = None
        self.fdoa_model = None
        
        if tdoa_model_path:
            try:
                self.tdoa_model = tf.keras.models.load_model(
                    tdoa_model_path,
                    custom_objects={'ResidualBlock': ResidualBlock}
                )
                print(f"✓ TDOA model loaded from {tdoa_model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load TDOA model: {e}")
        
        if fdoa_model_path:
            try:
                self.fdoa_model = tf.keras.models.load_model(
                    fdoa_model_path,
                    custom_objects={'ResidualBlock': ResidualBlock}
                )
                print(f"✓ FDOA model loaded from {fdoa_model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load FDOA model: {e}")
        
        # Error statistics (for ±3σ range calculation)
        self.tdoa_error_std = 5e-6  # Will be updated from validation
        self.fdoa_error_std = 50.0  # Will be updated from validation
    
    def normalize_tdoa(self, tdoa: float) -> float:
        """Normalize TDOA to [-1, 1] range"""
        return np.clip(tdoa / self.tdoa_max, -1.0, 1.0)
    
    def denormalize_tdoa(self, tdoa_norm: float) -> float:
        """Denormalize TDOA from [-1, 1] to seconds"""
        return float(tdoa_norm * self.tdoa_max)
    
    def normalize_fdoa(self, fdoa: float) -> float:
        """Normalize FDOA to [-1, 1] range"""
        return np.clip(fdoa / self.fdoa_max, -1.0, 1.0)
    
    def denormalize_fdoa(self, fdoa_norm: float) -> float:
        """Denormalize FDOA from [-1, 1] to Hz"""
        return float(fdoa_norm * self.fdoa_max)
    
    def estimate_tdoa(self, stft_feature: np.ndarray) -> Tuple[float, float]:
        """
        Estimate TDOA from STFT feature.
        
        Args:
            stft_feature: STFT magnitude (256, 256) or (1, 256, 256, 1)
        
        Returns:
            (tdoa_estimate, tdoa_uncertainty)
            - tdoa_estimate: TDOA in seconds
            - tdoa_uncertainty: ±3σ range (seconds)
        """
        if self.tdoa_model is None:
            raise ValueError("TDOA model not loaded!")
        
        # Ensure correct shape
        if stft_feature.ndim == 2:
            stft_feature = stft_feature[np.newaxis, ..., np.newaxis]
        elif stft_feature.ndim == 3:
            stft_feature = stft_feature[np.newaxis, ...]
        
        # Predict (normalized)
        tdoa_norm = self.tdoa_model.predict(stft_feature, verbose=0)[0, 0]
        
        # Denormalize
        tdoa_est = self.denormalize_tdoa(tdoa_norm)
        
        # Uncertainty: ±3σ (99.7% confidence from paper's Anderson-Darling test)
        tdoa_uncertainty = 3 * self.tdoa_error_std
        
        return tdoa_est, tdoa_uncertainty
    
    def estimate_fdoa(self, stft_feature: np.ndarray) -> Tuple[float, float]:
        """
        Estimate FDOA from STFT feature.
        
        Args:
            stft_feature: STFT magnitude (256, 256) or (1, 256, 256, 1)
        
        Returns:
            (fdoa_estimate, fdoa_uncertainty)
            - fdoa_estimate: FDOA in Hz
            - fdoa_uncertainty: ±3σ range (Hz)
        """
        if self.fdoa_model is None:
            raise ValueError("FDOA model not loaded!")
        
        # Ensure correct shape
        if stft_feature.ndim == 2:
            stft_feature = stft_feature[np.newaxis, ..., np.newaxis]
        elif stft_feature.ndim == 3:
            stft_feature = stft_feature[np.newaxis, ...]
        
        # Predict (normalized)
        fdoa_norm = self.fdoa_model.predict(stft_feature, verbose=0)[0, 0]
        
        # Denormalize
        fdoa_est = self.denormalize_fdoa(fdoa_norm)
        
        # Uncertainty: ±3σ
        fdoa_uncertainty = 3 * self.fdoa_error_std
        
        return fdoa_est, fdoa_uncertainty
    
    def update_error_statistics(self, tdoa_std: float, fdoa_std: float):
        """
        Update error statistics from validation set.
        
        Called after Anderson-Darling test confirms normal distribution.
        
        Args:
            tdoa_std: Standard deviation of TDOA errors (seconds)
            fdoa_std: Standard deviation of FDOA errors (Hz)
        """
        self.tdoa_error_std = tdoa_std
        self.fdoa_error_std = fdoa_std
        
        print(f"[STNN] Error statistics updated:")
        print(f"  TDOA: σ = {tdoa_std*1e6:.2f} μs (±3σ = {3*tdoa_std*1e6:.2f} μs)")
        print(f"  FDOA: σ = {fdoa_std:.2f} Hz (±3σ = {3*fdoa_std:.2f} Hz)")


def create_training_callbacks(model_name: str, checkpoint_dir: str = 'model') -> list:
    """
    Create training callbacks for STNN models.
    
    Args:
        model_name: Name of the model (tdoa/fdoa)
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        List of Keras callbacks
    """
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/stnn_{model_name}_best.keras",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10,
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=f'{checkpoint_dir}/logs/stnn_{model_name}',
            histogram_freq=1
        )
    ]
    
    return cbs


if __name__ == "__main__":
    # Test model creation
    print("="*60)
    print("STNN Model Architecture Test")
    print("="*60)
    
    # Build models
    tdoa_model = build_stnn_tdoa_model()
    fdoa_model = build_stnn_fdoa_model()
    
    print("\n[TDOA Model]")
    tdoa_model.summary()
    
    print("\n[FDOA Model]")
    fdoa_model.summary()
    
    # Test inference
    dummy_input = np.random.randn(1, 256, 256, 1).astype(np.float32)
    
    print("\n[Test Inference]")
    tdoa_out = tdoa_model.predict(dummy_input, verbose=0)
    fdoa_out = fdoa_model.predict(dummy_input, verbose=0)
    
    print(f"✓ TDOA output: {tdoa_out[0, 0]:.4f} (normalized)")
    print(f"✓ FDOA output: {fdoa_out[0, 0]:.4f} (normalized)")
    
    print("\n✓ STNN models created successfully!")