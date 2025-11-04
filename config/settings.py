# ======================================
# 📄 config/settings.py
# Purpose: Configuration for DETECTION-ONLY pipeline
# ======================================

import os
import numpy as np

# ======================================
# ⚙️ System Settings
# ======================================
USE_NTN_IF_AVAILABLE = True
GPU_INDEX = 0

# ======================================
# 📊 Dataset Parameters
# ======================================
NUM_SAMPLES_PER_CLASS = 100  # برای تست سریع -> کل 200 نمونه
NUM_SATELLITES_FOR_TDOA = 12
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# ======================================
# 🧪 Detection Settings
# ======================================
ABLATION_CONFIG = {
    'power_preserving_covert': False  # ⚠️ CRITICAL: Must be False for detection!
}

# Covert channel injection parameters
COVERT_AMP = 0.55  # Increased for CNN testing: ~5-7% power difference
                   # Once CNN works, reduce back to 0.45 for true covert testing

# Noise control (for robustness testing)
# 🔍 DEBUG Item 4: Temporarily disabled for testing (if AUC jumps → noise is the issue)
ADD_NOISE = False   # ⚠️ TEMPORARILY DISABLED for debugging

VALIDATION_SPLIT = 0.3  # 30% for test set

# Performance settings
DEFAULT_N_JOBS = 2  # محدود کن برای جلوگیری از overhead زیاد در محیط dev

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
#  Directory Management
# ======================================
def init_directories():
    for d in [DATASET_DIR, MODEL_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Directory ensured: {d}/")

# ======================================
# 🧮 Derived Parameters (Legacy - Deprecated)
# ======================================
# Note: COVERT_AMP is now defined directly above as a configurable parameter
# The following function is kept for backward compatibility but not used
def covert_scale_from_esno_db(esno_db):
    """Convert Es/N0 (dB) to amplitude (deprecated - use COVERT_AMP directly)."""
    return float(np.sqrt(10.0**(esno_db/10.0)))
