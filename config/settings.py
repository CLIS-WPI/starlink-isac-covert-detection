# ======================================
# 📄 config/settings.py
# Purpose: Centralized configuration (OPTIMIZED)
# ======================================

import os

# ======================================
# ⚙️ System Settings
# ======================================
USE_NTN_IF_AVAILABLE = True  # Set False for Rayleigh (faster testing)
GPU_INDEX = 0
DEFAULT_COVERT_ESNO_DB = 6.0

# ======================================
# 📊 Dataset Parameters
# ======================================
NUM_SAMPLES_PER_CLASS = 1500  # Total 3000 samples
NUM_SATELLITES_FOR_TDOA = 12
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# ======================================
# 🧠 Training Hyperparameters
# ======================================
TRAIN_EPOCHS = 30
TRAIN_BATCH = 64
LEARNING_RATE = 3e-4
VALIDATION_SPLIT = 0.2

# ======================================
# 🧪 Ablation Study Flags
# ======================================
ABLATION_CONFIG = {
    'use_spectrogram': True,
    'use_rx_features': True,
    'use_curvature_weights': True,
    'power_preserving_covert': True
}

# ======================================
# 🧮 RF/OFDM Parameters
# ======================================
CARRIER_FREQUENCY = 28e9
SUBCARRIER_SPACING = 60e3
FFT_SIZE = 64
NUM_OFDM_SYMBOLS = 10
CYCLIC_PREFIX_LENGTH = 8

SAT_ANTENNA = {
    "num_rows": 8,
    "num_cols": 8,
    "polarization": "dual",
    "polarization_type": "VH"
}

UT_ANTENNA = {
    "polarization": "single",
    "polarization_type": "V"
}

# NTN Geometry
SCENARIO_TOPOLOGY = "dur"
SAT_HEIGHT = 600e3
ELEVATION_ANGLE = 50.0

# MCS/LDPC
NUM_BITS_PER_SYMBOL = 4
CODERATE = 0.5
LDPC_K = 512
LDPC_N = 1024

# ======================================
# 📂 Directory Management
# ======================================
def init_directories():
    """Create necessary directories if they don't exist."""
    for d in [DATASET_DIR, MODEL_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory ensured: {d}/")

# ======================================
# 🧮 Derived Parameters
# ======================================
def get_experiment_name():
    """Generate unique experiment name based on ablation config."""
    parts = []
    if not ABLATION_CONFIG['use_spectrogram']:
        parts.append('no-spec')
    if not ABLATION_CONFIG['use_rx_features']:
        parts.append('no-rx')
    if not ABLATION_CONFIG['use_curvature_weights']:
        parts.append('no-curv')
    if not ABLATION_CONFIG['power_preserving_covert']:
        parts.append('no-pwr-pres')
    return '_'.join(parts) if parts else 'full'

EXPERIMENT_NAME = get_experiment_name()

def covert_scale_from_esno_db(esno_db):
    """Convert Es/N0 (dB) to linear amplitude scaling."""
    import numpy as np
    return float(np.sqrt(10.0**(esno_db/10.0)))

COVERT_AMP = covert_scale_from_esno_db(DEFAULT_COVERT_ESNO_DB)