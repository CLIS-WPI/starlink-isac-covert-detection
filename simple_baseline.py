#!/usr/bin/env python3
"""
Simple baseline: Logistic Regression on covert band features
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

print("="*70)
print("SIMPLE BASELINE: Logistic Regression on Covert Band")
print("="*70)

# Load dataset
d = pickle.load(open('dataset/dataset_samples500_sats12.pkl', 'rb'))
rx_grids = np.squeeze(d['rx_grids'], axis=1)  # (1000, 10, 64)
labels = d['labels']

print(f"\nDataset: {len(labels)} samples ({np.sum(labels==0)} benign, {np.sum(labels==1)} attack)")

# Extract features from covert band (subcarriers 0-15)
def extract_features(grids):
    """Extract simple features from covert band"""
    # Average magnitude over OFDM symbols for each subcarrier
    mag = np.abs(grids)  # (N, 10, 64)
    
    # Features: average magnitude of covert band (0-15)
    covert_band_mag = mag[:, :, :16]  # (N, 10, 16)
    
    # Simple features:
    features = []
    
    # 1. Mean magnitude per subcarrier in covert band
    feat_mean = np.mean(covert_band_mag, axis=1)  # (N, 16)
    features.append(feat_mean)
    
    # 2. Max magnitude per subcarrier
    feat_max = np.max(covert_band_mag, axis=1)  # (N, 16)
    features.append(feat_max)
    
    # 3. Std magnitude per subcarrier
    feat_std = np.std(covert_band_mag, axis=1)  # (N, 16)
    features.append(feat_std)
    
    # Concatenate all features
    X = np.concatenate(features, axis=1)  # (N, 48)
    return X

print("\nExtracting features...")
X = extract_features(rx_grids)
print(f"Feature shape: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"Train: {len(y_train)} samples")
print(f"Test:  {len(y_test)} samples")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nTest AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))

if auc > 0.70:
    print("✅ SUCCESS: Simple baseline works!")
    print("   → Problem is CNN architecture, not data!")
    print("   → Consider simpler CNN or different architecture")
elif auc > 0.60:
    print("⚠️  MARGINAL: Baseline shows some signal")
    print("   → Pattern exists but weak")
    print("   → CNN might need better features or more data")
else:
    print("❌ FAILED: Even baseline cannot detect")
    print("   → Pattern may not be learnable")
    print("   → Need to increase COVERT_AMP or change injection strategy")

print("="*70)
