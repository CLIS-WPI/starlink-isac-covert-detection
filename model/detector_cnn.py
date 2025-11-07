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
        
        # 🔧 FIX: Store normalization statistics (computed only on train data)
        self.norm_mean_ofdm = None
        self.norm_std_ofdm = None
        self.norm_mean_csi_real = None
        self.norm_std_csi_real = None
        self.norm_mean_csi_imag = None
        self.norm_std_csi_imag = None
        
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
        
        # 🔧 OPTIMIZATION: Add spatial attention for subcarrier pattern detection
        # This helps focus on frequencies where covert injection occurs
        attention_conv = layers.Conv2D(1, (1, 1), activation='sigmoid', 
                                       name=f"{name_prefix}_attention")(x)
        x = layers.Multiply(name=f"{name_prefix}_attn_mult")([x, attention_conv])
        
        # 🔧 OPTIMIZATION: Add one more conv block for deeper pattern learning
        x = layers.Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(0.001),
                         name=f"{name_prefix}_conv4")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn4")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act4")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name=f"{name_prefix}_drop4")(x)
        
        x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        
        return inputs, x
    
    def _build_csi_encoder(self, csi_shape, name_prefix="csi"):
        """
        Build encoder for CSI (Channel State Information) with CNN branch.
        
        🔧 IMPROVED: Use small CNN blocks instead of raw dense layers.
        This helps extract spatial/frequency patterns from CSI.
        """
        inputs = keras.Input(shape=csi_shape, name=f"{name_prefix}_input")
        
        x = inputs
        
        # Expand dims if needed for CNN processing
        if len(csi_shape) == 1:  # Flatten CSI vector
            # Reshape to 2D for CNN: (features, 1)
            x = layers.Reshape((csi_shape[0], 1, 1))(x)
        elif len(csi_shape) == 2:  # (symbols, subcarriers) or (features, channels)
            # Add channel dimension: (symbols, subcarriers, 1)
            if csi_shape[-1] == 2:  # Already has real/imag channels
                pass  # Keep as is
            else:
                x = layers.Reshape((*csi_shape, 1))(x)
        elif len(csi_shape) == 3:  # (symbols, subcarriers, channels)
            pass  # Already in correct format
        
        # 🔧 IMPROVED: Small CNN blocks for spatial/frequency pattern extraction
        # Conv Block 1: Extract local patterns
        if len(x.shape) >= 4:  # Has spatial dimensions
            x = layers.Conv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.001),
                            name=f"{name_prefix}_conv1")(x)
            x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
            x = layers.Activation('relu', name=f"{name_prefix}_act1")(x)
            x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
            
            # Conv Block 2: Deeper patterns
            x = layers.Conv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularizers.l2(0.001),
                            name=f"{name_prefix}_conv2")(x)
            x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
            x = layers.Activation('relu', name=f"{name_prefix}_act2")(x)
            x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        else:
            # Fallback to dense if not spatial
            x = layers.Flatten(name=f"{name_prefix}_flatten")(x)
            x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                            name=f"{name_prefix}_dense1")(x)
            x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
            x = layers.Activation('relu', name=f"{name_prefix}_act1")(x)
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop1")(x)
        
        # Final dense layer
        x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                        name=f"{name_prefix}_dense_final")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn_final")(x)
        x = layers.Activation('relu', name=f"{name_prefix}_act_final")(x)
        
        return inputs, x
    
    def _build_model(self, ofdm_shape, csi_shape=None):
        """
        Build complete CNN model (with or without CSI fusion).
        """
        if self.use_csi and csi_shape is not None:
            # Multi-modal fusion: OFDM + CSI
            ofdm_input, ofdm_features = self._build_ofdm_encoder(ofdm_shape)
            csi_input, csi_features = self._build_csi_encoder(csi_shape)
            
            # 🔧 IMPROVED: Attention-based fusion instead of raw concatenation
            # Compute attention weights for feature fusion
            # Attention mechanism: learn which features are more important
            
            # Get feature dimensions
            ofdm_dim = ofdm_features.shape[-1]
            csi_dim = csi_features.shape[-1]
            
            # Compute attention scores (scalar per modality)
            # 🔧 FIX: Use unique names to avoid conflict with ofdm_encoder's attention layer
            ofdm_attn_score = layers.Dense(1, activation='sigmoid', name="fusion_ofdm_attention")(ofdm_features)
            csi_attn_score = layers.Dense(1, activation='sigmoid', name="fusion_csi_attention")(csi_features)
            
            # Normalize attention weights (softmax over modalities)
            attn_scores = layers.Concatenate(name="attn_concat")([ofdm_attn_score, csi_attn_score])
            attn_weights = layers.Softmax(name="attn_softmax")(attn_scores)
            
            # Apply attention weights (broadcast to feature dimensions)
            # Use Lambda layers for slicing
            ofdm_weight = layers.Lambda(lambda x: x[:, 0:1], name="ofdm_weight_slice")(attn_weights)
            csi_weight = layers.Lambda(lambda x: x[:, 1:2], name="csi_weight_slice")(attn_weights)
            
            # Scale features by attention weights
            ofdm_weighted = layers.Multiply(name="ofdm_weighted")([ofdm_features, ofdm_weight])
            csi_weighted = layers.Multiply(name="csi_weighted")([csi_features, csi_weight])
            
            # Concatenate weighted features
            combined = layers.Concatenate(name="fusion")([ofdm_weighted, csi_weighted])
            
            # Classification head with deeper layers
            x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001),
                           name="fusion_dense1")(combined)
            x = layers.BatchNormalization(name="fusion_bn1")(x)
            x = layers.Activation('relu', name="fusion_act1")(x)
            x = layers.Dropout(self.dropout_rate, name="fusion_drop1")(x)
            
            x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                           name="fusion_dense2")(x)
            x = layers.BatchNormalization(name="fusion_bn2")(x)
            x = layers.Activation('relu', name="fusion_act2")(x)
            x = layers.Dropout(self.dropout_rate, name="fusion_drop2")(x)
            
            x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001),
                           name="fusion_dense3")(x)
            x = layers.BatchNormalization(name="fusion_bn3")(x)
            x = layers.Activation('relu', name="fusion_act3")(x)
            x = layers.Dropout(self.dropout_rate, name="fusion_drop3")(x)
            
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
        
        # 🔍 DEBUG: Store original for comparison
        if not hasattr(self, '_debug_preprocess_count'):
            self._debug_preprocess_count = 0
        self._debug_preprocess_count += 1
        debug_mode = (self._debug_preprocess_count <= 3)
        
        if debug_mode:
            print(f"    [Preprocess Debug #{self._debug_preprocess_count}]")
            print(f"      Input shape: {X_grids.shape}")
            print(f"      Input magnitude range: [{np.min(np.abs(X_grids)):.6f}, {np.max(np.abs(X_grids)):.6f}]")
        
        # Extract magnitude and phase
        magnitude = np.abs(X_grids)  # (N, symbols, subcarriers)
        phase = np.angle(X_grids)    # (N, symbols, subcarriers)
        
        # Stack as channels: (N, symbols, subcarriers, 2)
        X_processed = np.stack([magnitude, phase], axis=-1).astype(np.float32)
        
        # 🔧 FIX: Use stored normalization stats (computed only on train) or compute if training
        # This prevents data leakage: test data should use train statistics
        if self.norm_mean_ofdm is not None and self.norm_std_ofdm is not None:
            # Use stored statistics (from training)
            mean_to_use = self.norm_mean_ofdm
            std_to_use = self.norm_std_ofdm
        else:
            # Compute from current data (only during training)
            mean_to_use = np.mean(X_processed[..., 0])
            std_to_use = np.std(X_processed[..., 0])
            # Store for later use (test/validation)
            self.norm_mean_ofdm = mean_to_use
            self.norm_std_ofdm = std_to_use
        
        if std_to_use > 0:
            X_processed[..., 0] = (X_processed[..., 0] - mean_to_use) / std_to_use
        
        if debug_mode:
            print(f"      After z-score: range [{np.min(X_processed[..., 0]):.6f}, {np.max(X_processed[..., 0]):.6f}]")
            print(f"      Using mean: {mean_to_use:.6f}, std: {std_to_use:.6f}")
            if self.norm_mean_ofdm is not None:
                print(f"      (Using stored train statistics)")
        
        # Phase is already in [-π, π], normalize to [-1, 1]
        X_processed[..., 1] /= np.pi
        
        return X_processed
    
    def _preprocess_csi(self, X_csi):
        """
        Preprocess CSI data with dual-channel (real/imag) input.
        
        🔧 FIX: Use real/imag channels instead of just magnitude
        This preserves phase information which is crucial for detection.
        
        Args:
            X_csi: CSI data (N, ...) - can be complex or real
        
        Returns:
            X_processed: Normalized CSI features with real/imag channels
        """
        if isinstance(X_csi, tf.Tensor):
            X_csi = X_csi.numpy()
        
        X_csi = np.squeeze(X_csi)
        
        # 🔧 FIX: Extract real and imaginary parts (dual-channel)
        # Instead of just magnitude, use both real and imag for better detection
        if np.iscomplexobj(X_csi):
            real_part = np.real(X_csi).astype(np.float32)
            imag_part = np.imag(X_csi).astype(np.float32)
        else:
            # If already real, use as real channel and zero for imag
            real_part = X_csi.astype(np.float32)
            imag_part = np.zeros_like(real_part)
        
        # Stack as channels: (N, ..., 2) for real/imag
        if real_part.ndim == 2:  # (N, features)
            X_processed = np.stack([real_part, imag_part], axis=-1)  # (N, features, 2)
        elif real_part.ndim == 3:  # (N, symbols, subcarriers)
            X_processed = np.stack([real_part, imag_part], axis=-1)  # (N, symbols, subcarriers, 2)
        else:
            # Flatten and reshape if needed
            real_flat = real_part.reshape(real_part.shape[0], -1)
            imag_flat = imag_part.reshape(imag_part.shape[0], -1)
            X_processed = np.stack([real_flat, imag_flat], axis=-1)  # (N, features, 2)
        
        # 🔧 FIX: Use stored normalization stats (computed only on train) or compute if training
        # This prevents data leakage: test data should use train statistics
        if (self.norm_mean_csi_real is not None and self.norm_std_csi_real is not None and
            self.norm_mean_csi_imag is not None and self.norm_std_csi_imag is not None):
            # Use stored statistics (from training)
            mean_real = self.norm_mean_csi_real
            std_real = self.norm_std_csi_real
            mean_imag = self.norm_mean_csi_imag
            std_imag = self.norm_std_csi_imag
        else:
            # Compute from current data (only during training)
            mean_real = np.mean(X_processed[..., 0])
            std_real = np.std(X_processed[..., 0])
            mean_imag = np.mean(X_processed[..., 1])
            std_imag = np.std(X_processed[..., 1])
            # Store for later use (test/validation)
            self.norm_mean_csi_real = mean_real
            self.norm_std_csi_real = std_real
            self.norm_mean_csi_imag = mean_imag
            self.norm_std_csi_imag = std_imag
        
        if std_real > 0:
            X_processed[..., 0] = (X_processed[..., 0] - mean_real) / std_real
        if std_imag > 0:
            X_processed[..., 1] = (X_processed[..., 1] - mean_imag) / std_imag
        
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
        # 🔧 FIX: Auto-compute class weights if not provided (handles imbalance automatically)
        if class_weight is None:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = {i: float(w) for i, w in zip(classes, class_weights)}
            print(f"  📊 Auto-computed class weights: {class_weight}")
        else:
            print(f"  📊 Using provided class weights: {class_weight}")
        
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
                print(f"   → Using auto-computed class weights to balance training")
        
        print(f"   Final class weights: {class_weight}")
        
        # 🔧 FIX: Preprocess train data FIRST to compute and store normalization stats
        # This ensures validation/test use train statistics (no data leakage)
        X_train_proc = self._preprocess_ofdm(X_train)
        
        # Auto-detect input shape
        if self.input_shape is None:
            self.input_shape = X_train_proc.shape[1:]  # (symbols, subcarriers, 2)
        
        # Preprocess CSI if using fusion (also computes train statistics)
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
        
        # 🔧 FIX: Preprocess validation data AFTER train (uses stored train statistics)
        validation_data = None
        if X_val is not None and y_val is not None:
            # At this point, normalization stats are already computed from train data
            # So validation will use train statistics (correct!)
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
    
    def find_optimal_threshold(self, X_val, y_val, X_csi_val=None, metric='f1'):
        """
        Find optimal threshold using validation set.
        
        Args:
            X_val: Validation OFDM grids
            y_val: Validation labels
            X_csi_val: Validation CSI data (optional)
            metric: 'f1', 'youden', or 'balanced'
        
        Returns:
            optimal_threshold: Best threshold value
        """
        probs = self.predict_proba(X_val, X_csi_val)
        
        if metric == 'youden':
            # Youden's J statistic: maximize TPR - FPR
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_val, probs)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
        elif metric == 'f1':
            # Maximize F1 score
            from sklearn.metrics import f1_score
            best_f1 = 0
            optimal_threshold = 0.5
            for threshold in np.arange(0.1, 0.9, 0.01):
                preds = (probs >= threshold).astype(int)
                f1 = f1_score(y_val, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    optimal_threshold = threshold
        else:  # 'balanced' or default
            # Balanced accuracy
            from sklearn.metrics import balanced_accuracy_score
            best_bal_acc = 0
            optimal_threshold = 0.5
            for threshold in np.arange(0.1, 0.9, 0.01):
                preds = (probs >= threshold).astype(int)
                bal_acc = balanced_accuracy_score(y_val, preds)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    optimal_threshold = threshold
        
        return optimal_threshold
    
    def evaluate(self, X_test, y_test, X_csi_test=None, threshold=None, X_val=None, y_val=None, X_csi_val=None):
        """
        Evaluate detector performance with optional threshold optimization.
        
        Args:
            X_test: Test OFDM grids
            y_test: Test labels
            X_csi_test: Test CSI data (optional, for fusion)
            threshold: Decision threshold (if None, will optimize on val set if provided)
            X_val: Validation OFDM grids (for threshold optimization)
            y_val: Validation labels (for threshold optimization)
            X_csi_val: Validation CSI data (optional, for threshold optimization)
        
        Returns:
            metrics: Dictionary with AUC, precision, recall, F1, threshold
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Get predictions
        probs = self.predict_proba(X_test, X_csi_test)
        
        # Optimize threshold if not provided and validation set available
        if threshold is None and X_val is not None and y_val is not None:
            print("  🔧 Optimizing threshold on validation set...")
            threshold = self.find_optimal_threshold(X_val, y_val, X_csi_val, metric='f1')
            print(f"  ✓ Optimal threshold: {threshold:.4f}")
        elif threshold is None:
            threshold = 0.5  # Default
        
        preds = (probs >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary', zero_division=0
        )
        
        metrics = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold
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
