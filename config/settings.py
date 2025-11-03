# ======================================
# 📄 config/settings.py
# Purpose: Centralized configuration (OPTIMIZED for LEO FDOA)
# ======================================

import os
import numpy as np

# ======================================
# ⚙️ System Settings
# ======================================
USE_NTN_IF_AVAILABLE = True
GPU_INDEX = 0
DEFAULT_COVERT_ESNO_DB = 20.0  # 🔧 Covert signal strength (higher = more detectable, lower = harder to detect)
                               # 15 dB gives good balance: spectral Cohen's d ~0.2-0.4

# ======================================
# 📊 Dataset Parameters
# ======================================
NUM_SAMPLES_PER_CLASS = 1500  # 🎯 PRODUCTION: 3000 total (1500+1500) for best quality
NUM_SATELLITES_FOR_TDOA = 12
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# TLE file path for SGP4-based constellation (optional)
TLE_PATH = "data/starlink.txt"  # Set to None if not using TLE-based orbit

# ======================================
# 🧠 Training Hyperparameters
# ======================================
TRAIN_EPOCHS = 100              # 🔧 Increased from 50 to 100 (with early stopping)
TRAIN_BATCH = 64
LEARNING_RATE = 5e-5            # 🔧 Reduced from 1e-4 to 5e-5 (slower, more stable)
VALIDATION_SPLIT = 0.2

# ======================================
# 🧪 Ablation Study Flags
# ======================================
ABLATION_CONFIG = {
    'use_spectrogram': True,
    'use_rx_features': True,
    'use_curvature_weights': True,
    'power_preserving_covert': True  # 🔧 ENABLED: Keep power ratio ~1.0 for realistic covert channel
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

# ======================================
# 🛰 LEO Satellite Dynamics (NEW)
# ======================================
# Orbital velocity for LEO at ~600 km altitude ≈ 7.56 km/s
LEO_ORBITAL_VELOCITY_MPS = 7560.0

# Radial component relative to ground terminal depends on elevation angle
LEO_RADIAL_VELOCITY_MPS = LEO_ORBITAL_VELOCITY_MPS * np.cos(np.deg2rad(ELEVATION_ANGLE))

# Maximum Doppler shift per satellite (for 28 GHz carrier)
LEO_MAX_DOPPLER_HZ = (LEO_RADIAL_VELOCITY_MPS / 3e8) * CARRIER_FREQUENCY

# ======================================
# 📡 MCS/LDPC
# ======================================
NUM_BITS_PER_SYMBOL = 4
CODERATE = 0.5
LDPC_K = 512
LDPC_N = 1024

# ======================================
# 📍 Localization Settings
# ======================================
USE_FDOA = True                     # ✅ enable FDOA refinement
USE_TDOA = True
FDOA_USE_SAT_VELOCITY = True        # ✅ enable use of satellite velocities
FDOA_MAX_DOPPLER_HZ = LEO_MAX_DOPPLER_HZ
FDOA_ELEVATION_ANGLE_DEG = ELEVATION_ANGLE

RESIDUAL_CNN_PATH = "model/localization_residual_cnn.keras"

# ======================================
# 🚀 STNN Configuration
# ======================================
USE_STNN_LOCALIZATION = True
STNN_TDOA_MODEL_PATH = "model/stnn_tdoa_best.keras"
STNN_FDOA_MODEL_PATH = "model/stnn_fdoa_best.keras"
STNN_ERROR_STATS_PATH = "model/stnn_error_stats.pkl"

# STNN Feature Extraction
STNN_STFT_NPERSEG = 256
STNN_STFT_OUTPUT_SHAPE = (256, 256)

# STNN Training
STNN_EPOCHS_TDOA = 50
STNN_EPOCHS_FDOA = 50
STNN_BATCH_SIZE = 32
STNN_USE_MULTI_GPU = True

# ======================================
# 📂 Directory Management
# ======================================
def init_directories():
    for d in [DATASET_DIR, MODEL_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory ensured: {d}/")

# ======================================
# 🧮 Derived Parameters
# ======================================
def get_experiment_name():
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
    return float(np.sqrt(10.0**(esno_db/10.0)))

COVERT_AMP = covert_scale_from_esno_db(DEFAULT_COVERT_ESNO_DB)
