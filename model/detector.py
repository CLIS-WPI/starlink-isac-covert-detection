# ======================================
# 📄 model/detector.py
# Purpose: Dual-input CNN detector (H100 optimized with FP32 output head)
# FIXED: Safe dataset creation with empty-set guards
# ======================================

# ======================================
# 📄Temperature Scaling
# ======================================

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, mixed_precision, callbacks
from config.settings import *
import numpy as np
from sklearn.model_selection import train_test_split


def build_dual_input_cnn_h100():
    """
    Build H100-optimized dual-input CNN with STRONGER REGULARIZATION.
    
    Returns:
        Model: Compiled Keras model (outputs logits)
    """
    # Spectrogram branch
    if ABLATION_CONFIG.get('use_spectrogram', True):
        a_in = layers.Input(shape=(64, 64, 1), name="spectrogram")
        a = layers.Conv2D(32, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(a_in)  # ✅ 0.001 → 0.01
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.4)(a)  # ✅ 0.25 → 0.4
        
        a = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(a)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.4)(a)  # ✅ 0.25 → 0.4
        
        a = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(a)
        a = layers.BatchNormalization()(a)
        a = layers.MaxPooling2D(2)(a)
        a = layers.Dropout(0.5)(a)  # ✅ 0.3 → 0.5
        
        a = layers.Conv2D(256, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(a)
        a = layers.BatchNormalization()(a)
        a = layers.GlobalAveragePooling2D()(a)
        a = layers.Dropout(0.5)(a)  # ✅ NEW
    else:
        a_in = layers.Input(shape=(64, 64, 1), name="spectrogram")
        a = layers.Flatten()(a_in)
        a = layers.Lambda(lambda x: x * 0)(a)
    
    # RX features branch
    if ABLATION_CONFIG.get('use_rx_features', True):
        b_in = layers.Input(shape=(8, 8, 3), name="rx_features")
        b = layers.Conv2D(32, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(b_in)  # ✅ NEW
        b = layers.BatchNormalization()(b)  # ✅ NEW
        b = layers.MaxPooling2D(2)(b)
        b = layers.Dropout(0.3)(b)  # ✅ NEW
        
        b = layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(b)
        b = layers.BatchNormalization()(b)  # ✅ NEW
        b = layers.MaxPooling2D(2)(b)
        b = layers.Dropout(0.3)(b)  # ✅ NEW
        
        b = layers.Conv2D(128, 3, padding='same', activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(b)
        b = layers.BatchNormalization()(b)  # ✅ NEW
        b = layers.GlobalAveragePooling2D()(b)
        b = layers.Dropout(0.4)(b)  # ✅ NEW
    else:
        b_in = layers.Input(shape=(8, 8, 3), name="rx_features")
        b = layers.Flatten()(b_in)
        b = layers.Lambda(lambda x: x * 0)(b)
    
    # Merge and classify
    x = layers.Concatenate()([a, b])
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # ✅ NEW
    x = layers.BatchNormalization()(x)  # ✅ NEW
    x = layers.Dropout(0.6)(x)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # ✅ NEW
    x = layers.BatchNormalization()(x)  # ✅ NEW
    x = layers.Dropout(0.5)(x)  # ✅ 0.3 → 0.5
    
    # ✅ Output logits (no sigmoid!)
    out = layers.Dense(1, dtype='float32', name='logits')(x)
    
    model = Model([a_in, b_in], out)
    
    opt = optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-7)
    
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', from_logits=True)
        ],
        jit_compile=True,
        steps_per_execution=128
    )
    
    return model


def find_optimal_temperature(model, Xs_val, Xr_val, y_val, max_iter=100):
    """
    Find optimal temperature using validation set.
    
    Args:
        model: Trained model (outputs logits)
        Xs_val, Xr_val, y_val: Validation data
        max_iter: Number of optimization iterations
    
    Returns:
        float: Optimal temperature T
    """
    print("\n[Calibration] Finding optimal temperature...")
    
    # Get logits on validation set
    val_ds = tf.data.Dataset.from_tensor_slices(((Xs_val, Xr_val), y_val))
    val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)
    
    val_logits = model.predict(val_ds, verbose=0)
    
    # ✅ FIX: Flatten logits to match y_val shape
    val_logits = val_logits.ravel()  # Convert (253, 1) to (253,)
    
    y_val_tensor = tf.constant(y_val, dtype=tf.float32)
    
    # Initialize temperature
    temperature = tf.Variable(1.0, dtype=tf.float32, name='temperature')
    
    def compute_loss():
        scaled_logits = val_logits / temperature
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_val_tensor, scaled_logits, from_logits=True
            )
        )
        return loss
    
    # Optimize temperature
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    for i in range(max_iter):
        optimizer.minimize(compute_loss, var_list=[temperature])
        if (i + 1) % 20 == 0:
            loss = compute_loss().numpy()
            print(f"  Iter {i+1}/{max_iter}: T={temperature.numpy():.4f}, Loss={loss:.4f}")
    
    T_optimal = float(temperature.numpy())
    print(f"✓ Optimal Temperature: {T_optimal:.4f}")
    
    return T_optimal


def train_detector(Xs_tr, Xr_tr, y_tr, Xs_te, Xr_te, y_te):
    """
    Train the dual-input CNN detector with temperature scaling.
    
    Args:
        Xs_tr, Xr_tr, y_tr: Training data
        Xs_te, Xr_te, y_te: Test data
    
    Returns:
        tuple: (model, history, temperature)
    """
    print("\n[Model] Building H100-optimized dual-input CNN...")
    
    # Enable mixed precision + XLA
    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)
    
    # Convert to float32
    Xs_tr, Xr_tr, y_tr = map(np.float32, [Xs_tr, Xr_tr, y_tr])
    Xs_te, Xr_te, y_te = map(np.float32, [Xs_te, Xr_te, y_te])
    
    # ✅ NEW: Split training into train + validation for temperature calibration
    print(f"\n=== DATA SPLIT FOR CALIBRATION ===")
    Xs_train, Xs_val, Xr_train, Xr_val, y_train, y_val = train_test_split(
        Xs_tr, Xr_tr, y_tr,
        test_size=0.1,  # 10% for calibration
        stratify=y_tr,
        random_state=42
    )
    
    print(f"Train set (for training): {len(Xs_train)} samples")
    print(f"Val set (for calibration): {len(Xs_val)} samples")
    print(f"Test set (final eval): {len(Xs_te)} samples")
    
    # Build datasets
    AUTOTUNE = tf.data.AUTOTUNE
    
    def make_ds(Xs, Xr, y, batch, shuffle=False, drop_remainder=True):
        ds = tf.data.Dataset.from_tensor_slices(((Xs, Xr), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(y), 4096))
        ds = ds.batch(batch, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    
    train_ds = make_ds(Xs_train, Xr_train, y_train, TRAIN_BATCH, shuffle=True, drop_remainder=True)
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
    
    # Train
    print(f"\n[Model] Training for {TRAIN_EPOCHS} epochs...")
    
    if len(Xs_te) > 0:
        hist = model.fit(
            train_ds,
            epochs=TRAIN_EPOCHS,
            validation_data=test_ds,
            callbacks=cbs,
            verbose=1
        )
    else:
        hist = model.fit(
            train_ds,
            epochs=TRAIN_EPOCHS,
            callbacks=None,
            verbose=1
        )
    
    print("\n=== TRAINING SUMMARY ===")
    if 'val_auc' in hist.history:
        best_ep = 1 + int(np.argmax(hist.history['val_auc']))
        print(f"Best epoch: {best_ep}")
        print(f"Best val_auc: {np.max(hist.history['val_auc']):.4f}")
        print(f"Final train_acc: {hist.history['accuracy'][-1]:.4f}")
    
    # ✅ NEW: Temperature calibration
    temperature = find_optimal_temperature(model, Xs_val, Xr_val, y_val)
    
    return model, hist, temperature


def evaluate_detector(model, Xs_te, Xr_te, y_te, temperature=1.0):
    """
    Evaluate the trained detector with calibrated probabilities.
    
    Args:
        model: Trained Keras model (outputs logits)
        Xs_te, Xr_te, y_te: Test data
        temperature: Calibration temperature (from training)
    
    Returns:
        tuple: (y_prob_calibrated, best_thr, f1_scores, thresholds)
    """
    from sklearn.metrics import precision_recall_curve, f1_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    
    print("\n[Eval] Evaluating on test set...")
    
    # Build test dataset
    test_ds = tf.data.Dataset.from_tensor_slices(((Xs_te, Xr_te), y_te))
    test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Get logits
    test_logits = model.predict(test_ds, verbose=0)
    
    # ✅ FIX: Flatten logits to match y_te shape
    test_logits = test_logits.ravel()  # Convert (474, 1) to (474,)
    
    # ✅ Apply temperature scaling
    calibrated_logits = test_logits / temperature
    y_prob_calibrated = tf.nn.sigmoid(calibrated_logits).numpy().ravel()
    
    # Evaluate metrics
    loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            tf.constant(y_te, dtype=tf.float32),
            calibrated_logits,
            from_logits=True
        )
    ).numpy()
    
    acc = np.mean((y_prob_calibrated > 0.5).astype(int) == y_te)
    
    from sklearn.metrics import roc_auc_score
    auc_val = roc_auc_score(y_te, y_prob_calibrated)
    
    print(f"Test Results → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc_val:.4f}")
    
    # ✅ Calibration metrics
    print(f"\n=== CALIBRATION METRICS (T={temperature:.4f}) ===")
    brier = brier_score_loss(y_te, y_prob_calibrated)
    prob_true, prob_pred = calibration_curve(y_te, y_prob_calibrated, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    
    # Find best threshold
    p, r, t = precision_recall_curve(y_te, y_prob_calibrated)
    f1_scores = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    best_idx = np.argmax(f1_scores)
    best_thr = t[best_idx] if best_idx < len(t) else 0.5
    
    print(f"\n=== THRESHOLD TUNING ===")
    print(f"Default threshold (0.5): F1 = {f1_score(y_te, (y_prob_calibrated > 0.5).astype(int)):.4f}")
    print(f"Optimized threshold: {best_thr:.4f}")
    print(f"  Best F1 score: {f1_scores[best_idx]:.4f}")
    print(f"  Precision: {p[best_idx]:.4f}")
    print(f"  Recall: {r[best_idx]:.4f}")
    
    return y_prob_calibrated, best_thr, f1_scores, t