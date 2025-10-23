# ======================================
# 📄 model/detector.py
# Purpose: Dual-input CNN detector (H100 optimized with FP32 output head)
# FIXED: Safe dataset creation with empty-set guards
# ======================================

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, mixed_precision, callbacks
from config.settings import *
import numpy as np


def build_dual_input_cnn_h100():
    """
    Build H100-optimized dual-input CNN with exact architecture from original.
    
    Returns:
        Model: Compiled Keras model
    """
    # Spectrogram branch
    if ABLATION_CONFIG.get('use_spectrogram', True):
        a_in = layers.Input(shape=(64, 64, 1), name="spectrogram")
        a = layers.Conv2D(32, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(a_in)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.25)(a)
        
        a = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.25)(a)
        
        a = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.3)(a)
        
        a = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
        a = layers.BatchNormalization()(a)
        a = layers.GlobalAveragePooling2D()(a)
    else:
        a_in = layers.Input(shape=(64, 64, 1), name="spectrogram")
        a = layers.Flatten()(a_in)
        a = layers.Lambda(lambda x: x * 0)(a)
    
    # RX features branch
    if ABLATION_CONFIG.get('use_rx_features', True):
        b_in = layers.Input(shape=(8, 8, 3), name="rx_features")
        b = layers.Conv2D(32, 3, padding='same', activation='relu')(b_in)
        b = layers.MaxPooling2D(2)(b)
        b = layers.Conv2D(64, 3, padding='same', activation='relu')(b)
        b = layers.MaxPooling2D(2)(b)
        b = layers.Conv2D(128, 3, padding='same', activation='relu')(b)
        b = layers.GlobalAveragePooling2D()(b)
    else:
        b_in = layers.Input(shape=(8, 8, 3), name="rx_features")
        b = layers.Flatten()(b_in)
        b = layers.Lambda(lambda x: x * 0)(b)
    
    # Merge and classify
    x = layers.Concatenate()([a, b])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model([a_in, b_in], out)
    
    opt = optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-7)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
        jit_compile=True,
        steps_per_execution=128
    )
    
    return model


def train_detector(Xs_tr, Xr_tr, y_tr, Xs_te, Xr_te, y_te):
    """
    Train the dual-input CNN detector with safe dataset creation.
    
    FIXED: Handles small test sets and empty validation gracefully.
    
    Args:
        Xs_tr, Xr_tr, y_tr: Training data
        Xs_te, Xr_te, y_te: Test data
    
    Returns:
        tuple: (model, history)
    """
    print("\n[Model] Building H100-optimized dual-input CNN...")
    
    # Enable mixed precision + XLA
    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    
    # Convert to float32
    Xs_tr, Xr_tr, y_tr = map(np.float32, [Xs_tr, Xr_tr, y_tr])
    Xs_te, Xr_te, y_te = map(np.float32, [Xs_te, Xr_te, y_te])
    
    # ✅ FIX: Add diagnostics
    print(f"\n=== DATASET DIAGNOSTICS ===")
    print(f"Train set: {len(Xs_tr)} samples")
    print(f"Test set:  {len(Xs_te)} samples")
    print(f"Batch size: {TRAIN_BATCH}")
    print(f"Xs_tr shape: {Xs_tr.shape}")
    print(f"Xr_tr shape: {Xr_tr.shape}")
    print(f"Xs_te shape: {Xs_te.shape}")
    print(f"Xr_te shape: {Xr_te.shape}")
    
    # Guard against empty train set
    if len(Xs_tr) == 0:
        raise ValueError("❌ Training set is empty! Cannot proceed.")
    
    # Warn if test set is too small
    if len(Xs_te) == 0:
        print("⚠️ WARNING: Test set is empty! Training without validation.")
    elif len(Xs_te) < TRAIN_BATCH:
        print(f"⚠️ WARNING: Test set ({len(Xs_te)}) < batch size ({TRAIN_BATCH})")
        print("   Using drop_remainder=False for validation to avoid empty dataset.")
    
    # Build datasets
    AUTOTUNE = tf.data.AUTOTUNE
    
    def make_ds(Xs, Xr, y, batch, shuffle=False, drop_remainder=True):
        """Create TF dataset with configurable drop_remainder."""
        ds = tf.data.Dataset.from_tensor_slices(((Xs, Xr), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(y), 4096))
        ds = ds.batch(batch, drop_remainder=drop_remainder)  # ✅ Now configurable
        ds = ds.prefetch(AUTOTUNE)
        return ds
    
    # Train dataset: keep drop_remainder=True for stable steps
    train_ds = make_ds(Xs_tr, Xr_tr, y_tr, TRAIN_BATCH, shuffle=True, drop_remainder=True)
    
    # Test dataset: use drop_remainder=False to keep all samples
    test_ds = make_ds(Xs_te, Xr_te, y_te, TRAIN_BATCH, shuffle=False, drop_remainder=False)
    
    # Build model
    model = build_dual_input_cnn_h100()
    
    # Callbacks
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/best_model.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # ✅ FIX: Train with or without validation
    print(f"\n[Model] Training for {TRAIN_EPOCHS} epochs...")
    
    if len(Xs_te) > 0:
        # Normal training with validation
        hist = model.fit(
            train_ds,
            epochs=TRAIN_EPOCHS,
            validation_data=test_ds,
            callbacks=cbs,
            verbose=1
        )
    else:
        # Training without validation (test set empty)
        print("⚠️ Training without validation (test set is empty)")
        hist = model.fit(
            train_ds,
            epochs=TRAIN_EPOCHS,
            callbacks=None,  # Remove validation-based callbacks
            verbose=1
        )
    
    print("\n=== TRAINING SUMMARY ===")
    if 'val_auc' in hist.history:
        best_ep = 1 + int(np.argmax(hist.history['val_auc']))
        print(f"Best epoch: {best_ep}")
        print(f"Best val_auc: {np.max(hist.history['val_auc']):.4f}")
        print(f"Final train_acc: {hist.history['accuracy'][-1]:.4f}")
    
    return model, hist


def evaluate_detector(model, Xs_te, Xr_te, y_te):
    """
    Evaluate the trained detector.
    
    Args:
        model: Trained Keras model
        Xs_te, Xr_te, y_te: Test data
    
    Returns:
        tuple: (y_prob, best_thr, f1_scores, thresholds)
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    print("\n[Eval] Evaluating on test set...")
    
    # Build test dataset
    test_ds = tf.data.Dataset.from_tensor_slices(((Xs_te, Xr_te), y_te))
    test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Evaluate
    loss, acc, auc_val = model.evaluate(test_ds, verbose=0)
    print(f"Test Results → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc_val:.4f}")
    
    # Get predictions
    y_prob = model.predict(test_ds, verbose=0).ravel()
    
    # Find best threshold
    p, r, t = precision_recall_curve(y_te, y_prob)
    f1_scores = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    best_idx = np.argmax(f1_scores)
    best_thr = t[best_idx] if best_idx < len(t) else 0.5
    
    print(f"\n=== THRESHOLD TUNING ===")
    print(f"Default threshold (0.5): F1 = {f1_score(y_te, (y_prob > 0.5).astype(int)):.4f}")
    print(f"Optimized threshold: {best_thr:.4f}")
    print(f"  Best F1 score: {f1_scores[best_idx]:.4f}")
    print(f"  Precision: {p[best_idx]:.4f}")
    print(f"  Recall: {r[best_idx]:.4f}")
    
    return y_prob, best_thr, f1_scores, t