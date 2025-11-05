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
SEED = 42  # Random seed for reproducibility

# ======================================
# 📊 Dataset Parameters
# ======================================
NUM_SAMPLES_PER_CLASS = 500
                               # With augmentation, effective samples = 6000 per class
NUM_SATELLITES_FOR_TDOA = 12
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# ======================================
# 🧪 Detection Settings
# ======================================
ABLATION_CONFIG = {
    'power_preserving_covert': True  # ✅ برعکس شد - حالا CNN بهتر یاد می‌گیره!
}

# =======================================================
# 💡 Covert Injection Settings (Semi-Fixed Pattern)
# =======================================================

# Covert channel injection parameters
COVERT_AMP = 0.2  # 🎯 کاهش شدید - برای power preservation واقعی
                   # با این مقدار: power diff < 2% و pattern هنوز detectable
                   # بعد از AUC > 0.85 می‌توان به 0.1 کاهش داد

# 🎯 SEMI-FIXED PATTERN STRATEGY (for better CNN learning)
# Instead of fully random, use controlled patterns that CNN can learn
USE_SEMI_FIXED_PATTERN = True  # Enable semi-fixed injection pattern

# 🔬 ADVANCED FEATURES (Multi-modal Learning)
CSI_FUSION = True
USE_SPECTROGRAM = False  # ❌ خاموش - magnitude-only STFT از pattern info می‌افته
SPECTROGRAM_TYPE = "stft"      # Options: "stft", "mel", "both"
USE_PHASE_FEATURES = True      # 🆕 Extract phase and cyclostationary features
USE_RESIDUAL_CNN = True

# Contiguous band injection (more spectral signature)
NUM_COVERT_SUBCARRIERS = 16   # 🎯 Reduced from 32 to 16 for stronger per-subcarrier energy
BAND_SIZE = 8                  # 🎯 باند پیوسته کوچک (SUBBAND_SIZE)
BAND_START_OPTIONS = list(range(0, 48, 4))  # 🎯 12 موقعیت - بیشتر diversity

# Symbol pattern options (semi-fixed) - 6 patterns for more diversity
SYMBOL_PATTERN_OPTIONS = [
    [1, 3, 5, 7],           # الگوی ۱ (سمبل‌های فرد)
    [2, 4, 6, 8],           # الگوی ۲ (سمبل‌های زوج)
    [0, 1, 4, 5, 8, 9],     # الگوی ۳ (paired)
    [2, 3, 6, 7],           # الگوی ۴ (middle)
    [0, 1, 2, 3, 4],        # الگوی ۵ (first half)
    [5, 6, 7, 8, 9]         # الگوی ۶ (second half)
]
# Alias for compatibility
SYMBOL_PATTERNS = SYMBOL_PATTERN_OPTIONS
SUBBAND_SIZE = BAND_SIZE  # Alias for documentation consistency

# ⚠️ CRITICAL: Disable randomization when using semi-fixed pattern!
# Legacy randomization (ONLY used if USE_SEMI_FIXED_PATTERN = False)
RANDOMIZE_SUBCARRIERS = False  # ❌ غیرفعال - must be False for semi-fixed pattern to work
RANDOMIZE_SYMBOLS = False      # ❌ غیرفعال - must be False for semi-fixed pattern to work
MAX_SUBCARRIERS = 48          # 🎯 Limit randomization to first 48 (not all 64) for pattern consistency
MAX_SYMBOLS = 10              # 🎯 Total OFDM symbols available
NUM_INJECT_SYMBOLS = 7        # 🎯 How many symbols to inject covert signal into

# Noise control (for robustness testing)
ADD_NOISE = True   # 🔧 فعال‌سازی نویز برای واقع‌گرایی
NOISE_STD = 0.01  # 🎯 کاهش نویز برای یادگیری بهتر

# 🎯 ADVANCED TRAINING SETTINGS
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0         # Focus on hard examples
FOCAL_LOSS_ALPHA = 0.25        # Class weighting
USE_DATA_AUGMENTATION = True   # 🆕 Apply data augmentation
AUGMENTATION_FACTOR = 1        # Generate 2x more samples via augmentation

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
