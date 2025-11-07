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
# 🔒 Reproducibility & Evaluation Settings (Phase 0)
# ======================================
GLOBAL_SEED = 42  # Global random seed for reproducibility (used across all modules)
SEED = GLOBAL_SEED  # Alias for backward compatibility
N_FOLDS = 5  # Number of folds for cross-validation (if needed)
N_SEEDS = 3  # Number of random seeds for robustness testing (if needed)

# cuDNN determinism (for reproducible GPU operations)
CUDA_DETERMINISTIC = True  # Set to True for reproducibility (may reduce performance)

# ======================================
# 📊 Dataset Parameters
# ======================================
NUM_SAMPLES_PER_CLASS = 2000  # 10x larger dataset for robust training (total: 10000 samples)
                               # For paper: 5000 samples per class
                               # For testing: 500 samples is sufficient
NUM_SATELLITES_FOR_TDOA = 12
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# ======================================
# 🧪 Detection Settings
# ======================================
ABLATION_CONFIG = {
    'power_preserving_covert': True  # For paper: True (power-preserving)
}

# =======================================================
# 💡 Covert Injection Settings (Semi-Fixed Pattern)
# =======================================================

# Covert channel injection parameters
COVERT_AMP = 0.5  # For paper: 0.5 (power-preserving, realistic and detectable)
                   # Progression: 0.2→AUC=0.49, 0.3→AUC=0.54, 0.4→AUC=0.60
                   # 0.5 should give AUC ≥ 0.70 (minimum target)
                   # For testing/debugging: 0.7~0.9 (detectable pattern)

# 🎯 FIXED PATTERN STRATEGY (for consistent CNN learning)
# Use FIXED band position instead of semi-fixed for better detectability
USE_SEMI_FIXED_PATTERN = False  # Disabled - use fixed pattern instead of semi-fixed

# 🔬 ADVANCED FEATURES (Multi-modal Learning)
CSI_FUSION = True
USE_SPECTROGRAM = False  # Disabled - magnitude-only STFT loses pattern information
SPECTROGRAM_TYPE = "stft"      # Options: "stft", "mel", "both"
USE_PHASE_FEATURES = True      # 🆕 Extract phase and cyclostationary features
USE_RESIDUAL_CNN = True

# ====== NEW: Scenario & CSI controls ======
INSIDER_MODE = 'ground'  # 'sat' (downlink) | 'ground' (uplink->relay)
USE_REALISTIC_CSI = True      # Enable estimation from pilots (LS/LMMSE)
CSI_ESTIMATION = 'LS'         # 'LS' now, add 'LMMSE' later
POWER_PRESERVING_COVERT = True  # For paper: True (power-preserving, realistic)
                                 # For testing/debugging: False (detectable pattern)
MIN_ELEVATION_DEG = 10.0
NUM_SENSING_SATS = 12
USE_SGP4 = False              # Off by default; enable with TLE later

# Contiguous band injection (more spectral signature)
NUM_COVERT_SUBCARRIERS = 16   # 🎯 Reduced from 32 to 16 for stronger per-subcarrier energy
BAND_SIZE = 8                  # Small contiguous band (SUBBAND_SIZE)
BAND_START_OPTIONS = list(range(0, 48, 4))  # 12 positions - more diversity

# Symbol pattern options (semi-fixed) - 6 patterns for more diversity
SYMBOL_PATTERN_OPTIONS = [
    [1, 3, 5, 7],           # Pattern 1 (odd symbols)
    [2, 4, 6, 8],           # Pattern 2 (even symbols)
    [0, 1, 4, 5, 8, 9],     # Pattern 3 (paired)
    [2, 3, 6, 7],           # Pattern 4 (middle)
    [0, 1, 2, 3, 4],        # Pattern 5 (first half)
    [5, 6, 7, 8, 9]         # Pattern 6 (second half)
]
# Alias for compatibility
SYMBOL_PATTERNS = SYMBOL_PATTERN_OPTIONS
SUBBAND_SIZE = BAND_SIZE  # Alias for documentation consistency

# ⚠️ CRITICAL: Disable randomization for FIXED pattern!
# Legacy randomization settings (ONLY used if USE_SEMI_FIXED_PATTERN = False)
RANDOMIZE_SUBCARRIERS = False  # Disabled - use fixed band_start=0
RANDOMIZE_SYMBOLS = False      # Disabled - use fixed symbol pattern
RANDOMIZE_BAND_START = False   # Disabled - always band_start=0
RANDOMIZE_SYMBOL_PATTERN = False  # Disabled - always pattern [1,3,5,7]
MAX_SUBCARRIERS = 48          # 🎯 Limit randomization to first 48 (not all 64) for pattern consistency
MAX_SYMBOLS = 10              # 🎯 Total OFDM symbols available
NUM_INJECT_SYMBOLS = 7        # 🎯 How many symbols to inject covert signal into

# Noise control (for robustness testing)
ADD_NOISE = True   # Enable noise for realism
NOISE_STD = 0.01  # Reduced noise for better learning

# 🎯 ADVANCED TRAINING SETTINGS
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.5         # Increased from 2.0 to 2.5 (more focus on hard examples)
FOCAL_LOSS_ALPHA = 0.5         # Increased from 0.25 to 0.5 (better balance)
USE_DATA_AUGMENTATION = True   # 🆕 Apply data augmentation
AUGMENTATION_FACTOR = 1        # Generate 2x more samples via augmentation

# 🔧 OPTIMIZATION: Learning rate scheduling
USE_LEARNING_RATE_SCHEDULER = True
INITIAL_LR = 0.001             # Initial learning rate
LR_DECAY_FACTOR = 0.5          # Decay factor
LR_PATIENCE = 5                # Reduce LR if no improvement for N epochs
MIN_LR = 1e-6                  # Minimum learning rate

VALIDATION_SPLIT = 0.3  # 30% for test set

# Performance settings
DEFAULT_N_JOBS = 2  # Limited to prevent excessive overhead in dev environment

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
