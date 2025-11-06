#!/usr/bin/env python3
"""
🧠 CNN-BASED COVERT CHANNEL DETECTOR
====================================
Deep learning detector for subtle covert channels with optional CSI fusion.

Architecture Options:
1. CNN-only: Learns from OFDM tx_grids (magnitude + phase)
2. CNN+CSI: Multi-modal fusion of OFDM + channel state information

Designed for:
- Ultra-low power difference detection (< 1%)
- Subtle spectral anomaly detection
- Real covert channel scenarios

Usage:
    from model.detector_cnn import CNNDetector
    
    # CNN-only
    detector = CNNDetector(use_csi=False)
    detector.train(X_grids, y_labels)
    
    # CNN+CSI fusion
    detector = CNNDetector(use_csi=True)
    detector.train(X_grids, y_labels, X_csi=csi_data)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import pickle


class CNNDetector:
    """
    CNN-based covert channel detector with optional CSI fusion.
    
    Features:
    - Automatic feature learning from raw OFDM grids
    - Optional multi-modal fusion with CSI data
    - Handles ultra-subtle covert channels (< 1% power difference)
    - Regularization to prevent overfitting on small datasets
    """
    
    def __init__(self, use_csi=False, input_shape=None, csi_shape=None, 
                 learning_rate=0.001, dropout_rate=0.3, random_state=42,
                 use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25):
        """
        Initialize CNN detector.
        
        Args:
            use_csi: Whether to use CSI fusion (default: False)
            input_shape: Shape of OFDM grid input (auto-detected if None)
            csi_shape: Shape of CSI input (auto-detected if None)
            learning_rate: Adam optimizer learning rate
            dropout_rate: Dropout probability for regularization
            random_state: Random seed for reproducibility
            use_focal_loss: Use focal loss instead of binary crossentropy
            focal_gamma: Focal loss gamma (focus on hard examples)
            focal_alpha: Focal loss alpha (class weight)
        """
        self.use_csi = use_csi
        self.input_shape = input_shape
        self.csi_shape = csi_shape
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        self.model = None
        self.is_trained = False
        self.history = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_ofdm_encoder(self, input_shape, name_prefix="ofdm"):
        """
        Build CNN encoder for OFDM tx_grids.
        
        Processes complex-valued OFDM grids (magnitude + phase).
        Architecture designed for subtle pattern detection.
        """
        # Input: (batch, symbols, subcarriers, 2) for magnitude+phase
        # Or: (batch, symbols, subcarriers) if real-only
        inputs = keras.Input(shape=input_shape, name=f"{name_prefix}_input")
        
        x = inputs
        
        # Expand dims if needed (e.g., grayscale image)
        if len(input_shape) == 2:  # (symbols, subcarriers)
            x = layers.Reshape((*input_shape, 1))(x)
        
        # Conv Block 1: Detect local patterns
        x = layers.Conv2D(32, (3, 3), padding='same', 
                         kernel_regularizer=regularizers.l2(0.001),
                         name=f"{name_prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act1")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)
        x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
        
        # Conv Block 2: Higher-level features
        x = layers.Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.001),
                         name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act2")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)
        x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop2")(x)
        
        # Conv Block 3: Deep representations
        x = layers.Conv2D(128, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.001),
                         name=f"{name_prefix}_conv3")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act3")(x)
        x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        
        return inputs, x
    
    def _build_csi_encoder(self, csi_shape, name_prefix="csi"):
        """
        Build encoder for CSI (Channel State Information).
        
        Processes channel magnitude/phase patterns.
        """
        inputs = keras.Input(shape=csi_shape, name=f"{name_prefix}_input")
        
        x = inputs
        
        # Expand dims if needed
        if len(csi_shape) == 1:  # Flatten CSI vector
            x = layers.Reshape((csi_shape[0], 1))(x)
        
        # Dense processing for CSI
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                        name=f"{name_prefix}_dense1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act1")(x)
        x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
        
        x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                        name=f"{name_prefix}_dense2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act2")(x)
        
        x = layers.Flatten(name=f"{name_prefix}_flatten")(x)
        
        return inputs, x
    
    def _build_model(self, ofdm_shape, csi_shape=None):
        """
        Build complete CNN model (with or without CSI fusion).
        """
        if self.use_csi and csi_shape is not None:
            # Multi-modal fusion: OFDM + CSI
            ofdm_input, ofdm_features = self._build_ofdm_encoder(ofdm_shape)
            csi_input, csi_features = self._build_csi_encoder(csi_shape)
            
            # Fusion layer
            combined = layers.Concatenate(name="fusion")([ofdm_features, csi_features])
            
            # Classification head
            x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                           name="fusion_dense1")(combined)
            x = layers.BatchNormalization(name="fusion_bn1")(x)
            x = layers.Activation('relu', name="fusion_act1")(x)
            x = layers.Dropout(self.dropout_rate, name="fusion_drop1")(x)
            
            x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                           name="fusion_dense2")(x)
            x = layers.BatchNormalization(name="fusion_bn2")(x)
            x = layers.Activation('relu', name="fusion_act2")(x)
            x = layers.Dropout(self.dropout_rate, name="fusion_drop2")(x)
            
            outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
            
            model = models.Model(inputs=[ofdm_input, csi_input], outputs=outputs,
                               name="CNN_CSI_Detector")
        else:
            # CNN-only: OFDM grids
            ofdm_input, ofdm_features = self._build_ofdm_encoder(ofdm_shape)
            
            # Classification head
            x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                           name="head_dense1")(ofdm_features)
            x = layers.BatchNormalization(name="head_bn1")(x)
            x = layers.Activation('relu', name="head_act1")(x)
            x = layers.Dropout(self.dropout_rate, name="head_drop1")(x)
            
            x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                           name="head_dense2")(x)
            x = layers.BatchNormalization(name="head_bn2")(x)
            x = layers.Activation('relu', name="head_act2")(x)
            x = layers.Dropout(self.dropout_rate, name="head_drop2")(x)
            
            outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
            
            model = models.Model(inputs=ofdm_input, outputs=outputs,
                               name="CNN_Detector")
        
        # Compile model
        # Use Focal Loss if enabled (better for hard examples)
        if self.use_focal_loss:
            loss = BinaryFocalCrossentropy(
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
                from_logits=False  # We use sigmoid activation
            )
            print(f"  ✓ Using Focal Loss (gamma={self.focal_gamma}, alpha={self.focal_alpha})")
        else:
            loss = 'binary_crossentropy'
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _preprocess_ofdm(self, X_grids):
        """
        Preprocess OFDM grids: extract magnitude and phase.
        
        Args:
            X_grids: Complex OFDM grids (N, symbols, subcarriers) or similar
        
        Returns:
            X_processed: (N, symbols, subcarriers, 2) for magnitude+phase
        """
        # Convert to numpy if TensorFlow tensor
        if isinstance(X_grids, tf.Tensor):
            X_grids = X_grids.numpy()
        
        # Squeeze extra dimensions
        X_grids = np.squeeze(X_grids)
        
        # Ensure 3D: (N, symbols, subcarriers)
        if X_grids.ndim == 2:
            X_grids = X_grids[np.newaxis, :, :]
        
        # Extract magnitude and phase
        magnitude = np.abs(X_grids)  # (N, symbols, subcarriers)
        phase = np.angle(X_grids)    # (N, symbols, subcarriers)
        
        # Stack as channels: (N, symbols, subcarriers, 2)
        X_processed = np.stack([magnitude, phase], axis=-1).astype(np.float32)
        
        # 🔧 GLOBAL normalization instead of per-sample
        # This preserves relative differences between samples (critical for detection!)
        # Old: mag_max per sample → destroyed pattern
        # New: global max across ALL samples → preserves pattern
        global_mag_max = np.max(X_processed[..., 0])
        if global_mag_max > 0:
            X_processed[..., 0] /= global_mag_max
        
        # Phase is already in [-π, π], normalize to [-1, 1]
        X_processed[..., 1] /= np.pi
        
        return X_processed
    
    def _preprocess_csi(self, X_csi):
        """
        Preprocess CSI data.
        
        Args:
            X_csi: CSI data (N, ...) - can be complex or real
        
        Returns:
            X_processed: Normalized CSI features
        """
        if isinstance(X_csi, tf.Tensor):
            X_csi = X_csi.numpy()
        
        # If complex, extract magnitude
        if np.iscomplexobj(X_csi):
            X_csi = np.abs(X_csi)
        
        X_csi = np.squeeze(X_csi).astype(np.float32)
        
        # Normalize to zero mean, unit variance
        mean = np.mean(X_csi, axis=0, keepdims=True)
        std = np.std(X_csi, axis=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        
        X_processed = (X_csi - mean) / std
        
        return X_processed
    
    def train(self, X_train, y_train, X_csi_train=None,
              X_val=None, y_val=None, X_csi_val=None,
              epochs=50, batch_size=32, verbose=1, class_weight=None):
        """
        Train the CNN detector.
        
        Args:
            X_train: Training OFDM grids
            y_train: Training labels (0=benign, 1=attack)
            X_csi_train: Training CSI data (optional, for fusion)
            X_val: Validation OFDM grids (optional)
            y_val: Validation labels (optional)
            X_csi_val: Validation CSI data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity (0=silent, 1=progress, 2=epoch)
            class_weight: Dictionary of class weights {0: w0, 1: w1} for handling imbalance
                         Default: {0: 1.0, 1: 1.0} (balanced)
        
        Returns:
            history: Training history
        """
        # 🎯 Class weights: Default to balanced, can be adjusted if needed
        if class_weight is None:
            class_weight = {0: 1.0, 1: 1.0}
        
        # Check class balance and warn if needed
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"\n📊 Class distribution in training set:")
        print(f"   Class 0 (benign): {class_dist.get(0, 0)} samples")
        print(f"   Class 1 (attack): {class_dist.get(1, 0)} samples")
        
        if len(class_dist) == 2:
            imbalance_ratio = max(counts) / min(counts)
            if imbalance_ratio > 1.5:
                print(f"   ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.2f})")
                print(f"   Consider adjusting class_weight parameter")
        
        print(f"   Using class weights: {class_weight}")
        
        # Preprocess OFDM data
        X_train_proc = self._preprocess_ofdm(X_train)
        
        # Auto-detect input shape
        if self.input_shape is None:
            self.input_shape = X_train_proc.shape[1:]  # (symbols, subcarriers, 2)
        
        # Preprocess CSI if using fusion
        X_csi_train_proc = None
        if self.use_csi and X_csi_train is not None:
            X_csi_train_proc = self._preprocess_csi(X_csi_train)
            if self.csi_shape is None:
                self.csi_shape = X_csi_train_proc.shape[1:]
        
        # Build model
        self.model = self._build_model(self.input_shape, self.csi_shape)
        
        print("\n" + "="*70)
        print("🧠 CNN DETECTOR ARCHITECTURE")
        print("="*70)
        self.model.summary()
        print("="*70)
        
        # Prepare training data
        if self.use_csi and X_csi_train_proc is not None:
            train_data = [X_train_proc, X_csi_train_proc]
        else:
            train_data = X_train_proc
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_proc = self._preprocess_ofdm(X_val)
            if self.use_csi and X_csi_val is not None:
                X_csi_val_proc = self._preprocess_csi(X_csi_val)
                validation_data = ([X_val_proc, X_csi_val_proc], y_val)
            else:
                validation_data = (X_val_proc, y_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc' if validation_data else 'auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n🚀 Training CNN detector for {epochs} epochs...")
        print(f"   Training samples: {len(y_train)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Using CSI fusion: {self.use_csi}")
        print(f"   Class weights: {class_weight}")
        
        self.history = self.model.fit(
            train_data, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,  # 🎯 Apply class weights to prevent bias
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Print final metrics
        print(f"\n✅ Training complete!")
        if validation_data:
            final_auc = self.history.history['val_auc'][-1]
            print(f"   Final validation AUC: {final_auc:.4f}")
        else:
            final_auc = self.history.history['auc'][-1]
            print(f"   Final training AUC: {final_auc:.4f}")
        
        return self.history
    
    def predict(self, X_test, X_csi_test=None):
        """
        Predict covert channel presence.
        
        Args:
            X_test: Test OFDM grids
            X_csi_test: Test CSI data (optional, for fusion)
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Preprocess
        X_test_proc = self._preprocess_ofdm(X_test)
        
        if self.use_csi and X_csi_test is not None:
            X_csi_test_proc = self._preprocess_csi(X_csi_test)
            test_data = [X_test_proc, X_csi_test_proc]
        else:
            test_data = X_test_proc
        
        # Predict probabilities
        probs = self.model.predict(test_data, verbose=0)
        
        # Convert to binary predictions
        predictions = (probs >= 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X_test, X_csi_test=None):
        """
        Predict probabilities of covert channel presence.
        
        Args:
            X_test: Test OFDM grids
            X_csi_test: Test CSI data (optional, for fusion)
        
        Returns:
            probabilities: Predicted probabilities (0 to 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Preprocess
        X_test_proc = self._preprocess_ofdm(X_test)
        
        if self.use_csi and X_csi_test is not None:
            X_csi_test_proc = self._preprocess_csi(X_csi_test)
            test_data = [X_test_proc, X_csi_test_proc]
        else:
            test_data = X_test_proc
        
        # Predict probabilities
        probs = self.model.predict(test_data, verbose=0).flatten()
        
        return probs
    
    def evaluate(self, X_test, y_test, X_csi_test=None):
        """
        Evaluate detector performance.
        
        Args:
            X_test: Test OFDM grids
            y_test: Test labels
            X_csi_test: Test CSI data (optional, for fusion)
        
        Returns:
            metrics: Dictionary with AUC, precision, recall, F1
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Get predictions
        probs = self.predict_proba(X_test, X_csi_test)
        preds = (probs >= 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary', zero_division=0
        )
        
        metrics = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained! Cannot save.")
        
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
