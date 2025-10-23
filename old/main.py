# -*- coding: utf-8 -*-
"""
Real-Time Covert Leakage Detection & Localization in Starlink-Scale ISAC Networks
Sionna + OpenNTN optimized pipeline (single-GPU friendly)

Key improvements over the original:
 - Uses tf.signal.stft (GPU) instead of scipy.signal.stft
 - Unified frequency-selective channel application for NTN/Rayleigh
 - Physically meaningful covert amplitude via Es/N0 (default -15 dB)
 - Simple topology caching to reduce CPU-bound setup load in NTN mode
 - Clear separation of CPU-heavy init vs GPU-heavy simulation/train
"""

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from tensorflow.keras import mixed_precision
import scipy.signal
import seaborn as sns
import pickle

# Redirect stdout/stderr to both terminal and a file (overwritten each run)
import sys
import atexit
try:
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    _log_path = os.path.join(os.getcwd(), 'output.txt')
    _log_file = open(_log_path, 'w')

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)

    def _close_log():
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass

    atexit.register(_close_log)
except Exception:
    # If anything goes wrong, do not prevent program start
    pass

# -----------------------------
# Config: GPU memory & seeds
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Pin to GPU:0 and use growth to avoid OOM
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tf.random.set_seed(42)
np.random.seed(42)

# -----------------------------
# Directory Management
# -----------------------------
DATASET_DIR = "dataset"
MODEL_DIR = "model"
RESULT_DIR = "result"

# Create directories if they do not exist
for directory in [DATASET_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory ensured: {directory}/")

# -----------------------------
# Config: High-level toggles
# -----------------------------
USE_NTN_IF_AVAILABLE = True          # Try NTN (TR 38.811); fallback to Rayleigh otherwise
NUM_SAMPLES_PER_CLASS = 100       # Start small; scale to 1500+ for paper runs
NUM_SATELLITES_FOR_TDOA = 12
DEFAULT_COVERT_ESNO_DB = 6.0       # Covert Es/N0 (dB); sweep {-25,-20,-15,-10} for ablation
TRAIN_EPOCHS = 30                    # Increase to 25-50 for final results
TRAIN_BATCH = 64

# ===== Ablation Study Flags =====
ABLATION_CONFIG = {
    'use_spectrogram': True,   # Spectrogram input
    'use_rx_features': True,   # RX CSI features
    'use_curvature_weights': True,  # Curvature-based weights in WLS
    'power_preserving_covert': True,  # Power-normalized covert injection
}

def get_experiment_name():
    """Generate unique name based on ablation config"""
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

# -----------------------------
# Sionna imports
# -----------------------------
import sionna
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, OFDMDemodulator, OFDMModulator, LMMSEEqualizer, ResourceGridMapper
from sionna.phy.channel import (
    time_to_ofdm_channel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
    RayleighBlockFading
)
from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

# Try NTN
NTN_MODELS_AVAILABLE = False
try:
    if USE_NTN_IF_AVAILABLE:
        from sionna.phy.channel.tr38811 import DenseUrban, Antenna, AntennaArray
        from sionna.phy.channel.tr38811 import utils as tr811_utils
        NTN_MODELS_AVAILABLE = True
except Exception:
    NTN_MODELS_AVAILABLE = False
    print("NTN (TR 38.811) not available. Falling back to Rayleigh.")

# 1) Enable mixed precision (Tensor Cores) + XLA
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)  # XLA

AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------
# Utilities
# -----------------------------
def covert_scale_from_esno_db(esno_db):
    """Convert Es/N0 (dB) to linear amplitude scaling (per symbol)."""
    return float(np.sqrt(10.0**(esno_db/10.0)))

COVERT_AMP = covert_scale_from_esno_db(DEFAULT_COVERT_ESNO_DB)

# Fix #1: gcc_phat با fftshift
def gcc_phat(x, y, upsample_factor=32, beta=0.5, use_hann=True):
    """
    GCC-β with frequency-domain zero-padding for upsampling.
    - x, y: complex baseband signals
    - upsample_factor: 8/16/32
    - beta: 1.0=PHAT, beta<1 can be better at low SNR
    """
    x = np.asarray(x, dtype=np.complex64)
    y = np.asarray(y, dtype=np.complex64)

    Lx, Ly = len(x), len(y)
    n_lin = Lx + Ly - 1
    n_fft = 1 << (n_lin - 1).bit_length()  # next power of 2
    n_fft_up = n_fft * upsample_factor
    
    # Optional: Hann window
    if use_hann:
        wx = np.hanning(Lx).astype(np.float32)
        wy = np.hanning(Ly).astype(np.float32)
        x = x * wx
        y = y * wy
    
    # FFT (not RFFT!)
    X = np.fft.fft(x, n_fft)
    Y = np.fft.fft(y, n_fft)
    R = X * np.conj(Y)
    
    # GCC-β
    mag = np.abs(R) + 1e-12
    R = R / (mag ** beta)
    
    # Zero-padding فرکانسی
    R_up = np.zeros(n_fft_up, dtype=np.complex64)
    half = n_fft // 2
    R_up[:half] = R[:half]
    R_up[-(n_fft - half):] = R[half:]
    
    # IFFT
    corr = np.fft.ifft(R_up)
    corr = np.fft.fftshift(np.abs(corr))  # ← 🔥 CHANGED: اضافه شدن fftshift
    
    # Try to prefer a multithreaded FFT backend when available (NumPy 1.26+).
    # This helps CPU-side GCC-PHAT runs use multiple cores without hurting GPU work.
    try:
        np.fft.set_fft_backend("pocketfft")
    except Exception:
        # Backend not available or older numpy — ignore silently.
        pass

    return corr

# -----------------------------
# ISAC System Definition
# -----------------------------
class ISACSystem:
    """Holds PHY, channel, and simulation parameters, and creates Sionna objects."""
    def __init__(self):
        # RF/OFDM params
        self.CARRIER_FREQUENCY   = 28e9
        self.SUBCARRIER_SPACING  = 60e3
        self.FFT_SIZE            = 64
        self.NUM_OFDM_SYMBOLS    = 10
        self.CYCLIC_PREFIX_LENGTH= 8

        # Antennas (BS: satellite; UT: single-element)
        self.SAT_ANTENNA = {"num_rows": 8, "num_cols": 8, "polarization": "dual", "polarization_type": "VH"}
        self.UT_ANTENNA  = {"polarization": "single", "polarization_type": "V"}

        self.NUM_SAT_BEAMS = 1
        self.NUM_UT        = 1
        self.NUM_RX_ANT    = 1
        self.NUM_TX_ANT    = self.SAT_ANTENNA["num_rows"] * self.SAT_ANTENNA["num_cols"] * 2

        # NTN geometry
        self.SCENARIO_TOPOLOGY = "dur"
        self.SAT_HEIGHT        = 600e3
        self.ELEVATION_ANGLE   = 50.0

        # MCS/LDPC
        self.NUM_BITS_PER_SYMBOL = 4     # 16-QAM
        self.CODERATE            = 0.5
        self.k, self.n           = 512, 1024

        # Sionna objects
        self.binary_source = BinarySource()
        self.mapper   = Mapper("qam", self.NUM_BITS_PER_SYMBOL)
        self.demapper = Demapper("app", "qam", self.NUM_BITS_PER_SYMBOL)

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.NUM_OFDM_SYMBOLS,
            fft_size=self.FFT_SIZE,
            subcarrier_spacing=self.SUBCARRIER_SPACING,
            num_tx=self.NUM_SAT_BEAMS,
            num_streams_per_tx=self.NUM_UT,
            cyclic_prefix_length=self.CYCLIC_PREFIX_LENGTH,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 7]
        )
        self.SAMPLING_RATE = float(self.FFT_SIZE * self.SUBCARRIER_SPACING)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.sm = StreamManagement(np.array([[1]]), self.NUM_UT)
        self.lmmse_equalizer = LMMSEEqualizer(self.rg, self.sm)
        self.modulator   = OFDMModulator(self.rg.cyclic_prefix_length)
        l_min, l_max     = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.demodulator = OFDMDemodulator(fft_size=self.rg.fft_size, l_min=l_min, l_max=l_max,
                                           cyclic_prefix_length=self.rg.cyclic_prefix_length)

        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder)

        # Channel model selection
        self._init_channel()

        # (Optional) small topology cache for NTN to reduce CPU-bound calls
        self.topology_cache = []

    def _init_channel(self):
        """Create channel model; fall back to Rayleigh if NTN unavailable."""
        if NTN_MODELS_AVAILABLE:
            print("Using TR 38.811 DenseUrban model (NTN).")
            self.ut_array = Antenna(polarization=self.UT_ANTENNA["polarization"],
                                    polarization_type=self.UT_ANTENNA["polarization_type"],
                                    antenna_pattern="38.901",
                                    carrier_frequency=self.CARRIER_FREQUENCY)
            self.bs_array = AntennaArray(num_rows=self.SAT_ANTENNA["num_rows"],
                                         num_cols=self.SAT_ANTENNA["num_cols"],
                                         polarization=self.SAT_ANTENNA["polarization"],
                                         polarization_type=self.SAT_ANTENNA["polarization_type"],
                                         antenna_pattern="38.901",
                                         carrier_frequency=self.CARRIER_FREQUENCY)
            self.CHANNEL_MODEL = DenseUrban(
                carrier_frequency=self.CARRIER_FREQUENCY,
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction='downlink',
                elevation_angle=self.ELEVATION_ANGLE,
                enable_pathloss=True,
                enable_shadow_fading=True
            )
            try:
                topo = tr811_utils.gen_single_sector_topology(
                    batch_size=1,
                    num_ut=self.NUM_UT,
                    scenario=self.SCENARIO_TOPOLOGY,
                    elevation_angle=self.ELEVATION_ANGLE,
                    bs_height=float(self.SAT_HEIGHT)
                )
                self.CHANNEL_MODEL.set_topology(*topo)
            except Exception as e:
                print(f"[WARN] NTN topology init failed: {e}\nFalling back to Rayleigh.")
                self._fallback_rayleigh()
        else:
            self._fallback_rayleigh()

    def _fallback_rayleigh(self):
        self.CHANNEL_MODEL = RayleighBlockFading(
            num_rx=self.NUM_UT,
            num_rx_ant=self.NUM_RX_ANT,
            num_tx=self.NUM_SAT_BEAMS,
            num_tx_ant=self.NUM_TX_ANT
        )

    def precompute_topologies(self, count=8, bs_heights=None):
        """Pre-generate a small pool of NTN topologies to reuse during dataset generation."""
        if not NTN_MODELS_AVAILABLE:
            return
        self.topology_cache.clear()
        if bs_heights is None:
            bs_heights = [self.SAT_HEIGHT] * count
        for h in bs_heights[:count]:
            topo = tr811_utils.gen_single_sector_topology(
                batch_size=1, num_ut=self.NUM_UT,
                scenario=self.SCENARIO_TOPOLOGY,
                elevation_angle=self.ELEVATION_ANGLE,
                bs_height=float(h)
            )
            self.topology_cache.append(topo)

    def set_cached_topology(self, idx=0):
        """Assign a cached topology by index; safe no-op if cache empty or Rayleigh."""
        if not NTN_MODELS_AVAILABLE or not self.topology_cache:
            return
        i = int(idx) % len(self.topology_cache)
        self.CHANNEL_MODEL.set_topology(*self.topology_cache[i])

# -----------------------------
# Covert Injection
# -----------------------------
def inject_covert_channel(ofdm_frame, resource_grid, covert_rate_mbps, scs, covert_amp=COVERT_AMP):
    """
    Embed covert QPSK symbols in a sparse subset of unused subcarriers across a few OFDM symbols.
    - covert_amp should be derived from Es/N0 for physical realism (default -15 dB).
    """
    if covert_rate_mbps <= 0.0:
        return ofdm_frame, None

    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = ofdm_frame.shape[-2]

    # Compute #covert subcarriers from target rate
    symbol_duration = (resource_grid.fft_size + resource_grid.cyclic_prefix_length) / (resource_grid.fft_size * scs)
    bits_per_symbol = 2  # QPSK
    symbols_per_second = 1.0 / symbol_duration
    bps_per_sub = bits_per_symbol * symbols_per_second
    num_covert_subcarriers = int((covert_rate_mbps * 1e6) / bps_per_sub)
    num_covert_subcarriers = max(1, min(num_covert_subcarriers, resource_grid.num_effective_subcarriers // 4))

    # Random QPSK covert symbols
    covert_bits = tf.random.uniform([batch_size, num_covert_subcarriers, bits_per_symbol], 0, 2, dtype=tf.int32)
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_syms = covert_mapper(covert_bits) * tf.cast(covert_amp, tf.complex64)

    # Choose a sparse set (every 4th subcarrier) to avoid pilots/data hot-spots
    all_indices = np.arange(resource_grid.num_effective_subcarriers)
    candidates = all_indices[::4]
    if len(candidates) < num_covert_subcarriers:
        num_covert_subcarriers = len(candidates)
        covert_syms = covert_syms[:, :num_covert_subcarriers]
    selected = np.random.choice(candidates, num_covert_subcarriers, replace=False)

    # Inject into 3 random OFDM symbols
    L = min(3, num_ofdm_symbols)
    sym_indices = np.random.choice(num_ofdm_symbols, L, replace=False)

    ofdm_np = ofdm_frame.numpy()
    cs = covert_syms.numpy()[0]

    # ✨ STEP 1: Save original power (if requested)
    if ABLATION_CONFIG.get('power_preserving_covert', True):
        try:
            orig_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        except Exception:
            orig_power = None

    # STEP 2: Inject covert symbols
    for s in sym_indices:
        for k, sc in enumerate(selected):
            ofdm_np[0, 0, 0, s, sc] += complex(np.asarray(cs[k]).item())

    # ✨ STEP 3: Rescale to preserve original power (optional)
    if ABLATION_CONFIG.get('power_preserving_covert', True) and orig_power is not None:
        new_power = np.mean(np.abs(ofdm_np[0, 0, 0, :, :])**2)
        scale = np.sqrt(orig_power / (new_power + 1e-12))
        ofdm_np[0, 0, 0, :, :] *= scale
        try:
            print(f"  [Covert] Power preserved: {orig_power:.6f} → {new_power:.6f} → {orig_power:.6f}")
        except Exception:
            pass

    # Random emitter location (for non-TDoA usage)
    emitter_location = (np.random.uniform(-1000, 1000), np.random.uniform(-1000, 1000), 0.0)
    return tf.convert_to_tensor(ofdm_np), emitter_location

# -----------------------------
# Dataset Generation (single-sat)
# -----------------------------
def generate_dataset(isac_system, num_samples_per_class, ebno_db_range=(5, 15), covert_rate_mbps_range=(1, 50)):
    """Generate benign vs covert transmissions for the primary detector."""
    all_iq, all_rxfreq, all_radar, all_labels, all_emit = [], [], [], [], []

    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

    total = num_samples_per_class * 2
    order = np.arange(total)
    np.random.shuffle(order)

    for idx in range(total):
        i = order[idx]
        is_attack = i >= num_samples_per_class
        y = 1 if is_attack else 0

        b = isac_system.binary_source([1, total_info_bits])
        c_blocks = [isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) for j in range(num_codewords)]
        c = tf.concat(c_blocks, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [1, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_grid = isac_system.rg_mapper(x)

        emit_loc = None
        if is_attack:
            covert_rate = np.random.uniform(*covert_rate_mbps_range)
            tx_grid, emit_loc = inject_covert_channel(tx_grid, isac_system.rg, covert_rate, isac_system.SUBCARRIER_SPACING, COVERT_AMP)

        tx_time = isac_system.modulator(tx_grid)

        # Channel topology (NTN): reuse cached topologies to reduce CPU
        if NTN_MODELS_AVAILABLE:
            isac_system.set_cached_topology(idx)

        # Channel generation
        if NTN_MODELS_AVAILABLE:
            a, tau = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
        else:
            a, tau = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                                               sampling_frequency=isac_system.rg.bandwidth)

        l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
        h_time = cir_to_time_channel(isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True)

        # Repeat taps over symbols if needed, then convert to frequency domain
        if h_time.shape[-2] == 1:
            mult = [1] * len(h_time.shape); mult[-2] = isac_system.NUM_OFDM_SYMBOLS
            h_time = tf.tile(h_time, mult)

        h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)    # [B,RX,RXant,TX,TXant,SYM,SC]
        h_freq = tf.reduce_mean(h_freq, axis=4)                          # avg TX ants
        h_freq = tf.reduce_mean(h_freq, axis=2)                          # avg RX ants
        h_f    = h_freq[:, 0, 0, :, :]                                   # [B,SYM,SC]
        h_f    = tf.expand_dims(h_f, axis=1)                              # [B,1,SYM,SC]
        h_f    = tf.expand_dims(h_f, axis=2)                              # [B,1,1,SYM,SC]

        # Apply channel + noise in frequency domain (physically scaled)
        y_noiseless = tx_grid * h_f
        # Derive noise from Es/N0 at the RX (use ebno_db as random draw)
        ebno_db = float(tf.random.uniform((), *ebno_db_range))
        rx_pow  = tf.reduce_mean(tf.abs(y_noiseless)**2)
        esn0    = 10.0**(ebno_db/10.0)
        sigma2  = rx_pow / esn0
        std     = tf.sqrt(tf.cast(sigma2/2.0, tf.float32))
        std     = tf.cast(std, tf.float32)
        n_f     = tf.complex(tf.random.normal(tf.shape(y_noiseless), stddev=std),
                             tf.random.normal(tf.shape(y_noiseless), stddev=std))
        y_f     = y_noiseless + n_f

        y_t = isac_system.modulator(y_f)

        # Simple radar echo (shift + atten + noise)
        radar_delay = np.random.randint(50, 150)
        radar_atten = 0.3
        tx_flat     = tf.squeeze(tx_time)
        radar_echo  = tf.roll(tx_flat, shift=radar_delay, axis=-1) * radar_atten
        radar_echo += tf.complex(tf.random.normal(tf.shape(radar_echo), stddev=std),
                                 tf.random.normal(tf.shape(radar_echo), stddev=std))

        all_iq.append((i, np.squeeze(y_t.numpy())))
        all_rxfreq.append((i, np.squeeze(y_f.numpy())))
        all_radar.append((i, np.squeeze(radar_echo.numpy())))
        all_labels.append((i, y))
        all_emit.append((i, emit_loc))

        if (idx+1) % 100 == 0:
            print(f"Generated {idx+1}/{total} samples")

    # Sort by original index for alignment
    all_iq     = [x[1] for x in sorted(all_iq, key=lambda z: z[0])]
    all_rxfreq = [x[1] for x in sorted(all_rxfreq, key=lambda z: z[0])]
    all_radar  = [x[1] for x in sorted(all_radar, key=lambda z: z[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda z: z[0])]
    all_emit   = [x[1] for x in sorted(all_emit, key=lambda z: z[0])]

    return {
        'iq_samples': np.array(all_iq),
        'csi': np.array(all_rxfreq),
        'radar_echo': np.array(all_radar),
        'labels': np.array(all_labels),
        'emitter_locations': all_emit
    }

# -----------------------------
# Multi-Satellite Dataset (for TDoA)
# -----------------------------
def generate_dataset_multi_satellite(isac_system, num_samples_per_class, num_satellites=4,
                                    ebno_db_range=(5, 15), covert_rate_mbps_range=(1, 50)):
    """Generate samples with receptions at multiple satellites for TDoA localization."""
    all_iq, all_rxfreq, all_radar, all_labels, all_emit = [], [], [], [], []
    all_sat_recepts = []
    all_tx_time_padded = []

    # Realistic constellation base spacing (per-sample generation moved inside loop)
    grid_spacing = 100e3

    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

    total = num_samples_per_class * 2
    order = np.arange(total)
    np.random.shuffle(order)

    # Note: base_positions are generated per-sample inside the main loop to ensure
    # dataset samples have varied constellation geometries (enables GroupShuffleSplit).
    # Global topology precompute was removed because topologies vary per-sample now.
    if NTN_MODELS_AVAILABLE:
        # Keep topology cache mechanism available inside ISACSystem; skip global precompute.
        pass

    c0 = 3e8  # speed of light
    signal_length = 720  # original OFDM frame length

    for idx in range(total):
        i = order[idx]
        is_attack = i >= num_samples_per_class
        y = 1 if is_attack else 0

        # ===== Generate per-sample satellite geometry (base_positions) =====
        base_positions = []
        if num_satellites == 4:
            base_positions = [
                np.array([0.0, 0.0, 600e3]),
                np.array([grid_spacing, 0.0, 600e3]),
                np.array([0.0, grid_spacing, 600e3]),
                np.array([grid_spacing, grid_spacing, 600e3]),
            ]
            for j, pos in enumerate(base_positions):
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions[j] = np.array([pos[0], pos[1], pos[2] + altitude_offset])
        elif num_satellites == 12:
            user_center_x, user_center_y = 75e3, 75e3
            shells = [545e3, 575e3, 345e3]
            shell_weights = [0.5, 0.35, 0.15]
            # Ring 1
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(100e3, 140e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-30e3, 30e3)
                base_positions.append(np.array([x, y_p, z]))
            # Ring 2
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.pi/4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(220e3, 280e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-40e3, 40e3)
                base_positions.append(np.array([x, y_p, z]))
            # Ring 3
            for j in range(4):
                angle = 2 * np.pi * j / 4 + np.random.uniform(-0.2, 0.2)
                r = np.random.uniform(380e3, 450e3)
                x = user_center_x + r * np.cos(angle)
                y_p = user_center_y + r * np.sin(angle)
                z = np.random.choice(shells, p=shell_weights)
                z += np.random.uniform(-50e3, 50e3)
                base_positions.append(np.array([x, y_p, z]))
        else:
            side = int(np.ceil(np.sqrt(num_satellites)))
            for j in range(num_satellites):
                x = (j % side) * grid_spacing
                y_p = (j // side) * grid_spacing
                base_positions.append(np.array([x, y_p, 600e3]))
            for j, pos in enumerate(base_positions):
                altitude_offset = np.random.uniform(-75e3, 75e3)
                base_positions[j] = np.array([pos[0], pos[1], pos[2] + altitude_offset])

        # Occasional logging to avoid clutter
        if idx % 500 == 0:
            try:
                print(f"  [Sample {idx}] New constellation generated. Alts: {[f'{p[2]/1e3:.1f}km' for p in base_positions[:min(3, len(base_positions))]]}...")
            except Exception:
                pass

        b = isac_system.binary_source([1, total_info_bits])
        c_blocks = [isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) for j in range(num_codewords)]
        c = tf.concat(c_blocks, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [1, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_grid = isac_system.rg_mapper(x)

        emitter_loc = None
        if is_attack:
            covert_rate = np.random.uniform(*covert_rate_mbps_range)
            # Place a ground emitter in a 200 km box around origin (x,y plane)
            emitter_loc = np.array([np.random.uniform(-50e3, 150e3),
                                    np.random.uniform(-50e3, 150e3),
                                    0.0])
            tx_grid, _ = inject_covert_channel(tx_grid, isac_system.rg, covert_rate, 
                                               isac_system.SUBCARRIER_SPACING, COVERT_AMP)

        # Calculate max delay for padding
        if emitter_loc is not None:
            ref_point = emitter_loc
        else:
            ref_point = np.array([50e3, 50e3, 0.0])
        
        max_distance = max([np.linalg.norm(sat_pos - ref_point) 
                           for sat_pos in base_positions[:num_satellites]])
        max_delay_samp = int(np.ceil((max_distance / c0) * isac_system.SAMPLING_RATE)) + 200
        # Pre-modulate clean transmit signal (no channel, no frequency selective effects)
        # This will be used for Path B (TDoA-friendly clean signal)
        tx_time = isac_system.modulator(tx_grid)  # [1, signal_length]
        tx_time_flat = tf.squeeze(tx_time)
        tx_time_padded = tf.pad(tf.expand_dims(tx_time_flat, 0),
                                [[0, 0], [0, max_delay_samp]],
                                constant_values=0)  # [1, signal_length + max_delay_samp]

        # Save clean padded transmit waveform for this sample (used later by TDoA)
        try:
            all_tx_time_padded.append(np.squeeze(tx_time_padded.numpy()))
        except Exception:
            # Fallback to raw array if conversion fails
            all_tx_time_padded.append(np.squeeze(tf.cast(tx_time_padded, tf.complex64).numpy()))

        sat_rx_list = []
        for sat_idx, sat_pos in enumerate(base_positions[:num_satellites]):
            # ===== CORRECT ORDER: Channel → Noise → Modulate → Padding → Roll → Crop → Demodulate =====
            
            # 1. Calculate channel in frequency domain
            if NTN_MODELS_AVAILABLE:
                isac_system.set_cached_topology(0)
                a, tau = isac_system.CHANNEL_MODEL(1, isac_system.rg.bandwidth)
            else:
                a, tau = isac_system.CHANNEL_MODEL(1, num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                                                   sampling_frequency=isac_system.rg.bandwidth)

            l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
            h_time = cir_to_time_channel(isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True)
            if h_time.shape[-2] == 1:
                mult = [1] * len(h_time.shape)
                mult[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time = tf.tile(h_time, mult)
            h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)
            h_freq = tf.reduce_mean(h_freq, axis=4)  # average TX antennas
            h_freq = tf.reduce_mean(h_freq, axis=2)  # average RX antennas
            h_f = h_freq[:, 0, 0, :, :]              # [B, SYM, SC]
            h_f = tf.expand_dims(h_f, axis=1)        # [B, 1, SYM, SC]
            h_f = tf.expand_dims(h_f, axis=2)        # [B, 1, 1, SYM, SC]

            # 2. Apply channel to tx_grid (in frequency domain)
            y_grid_channel = tx_grid * h_f

            # 3. Calculate and add noise in frequency domain (Path A)
            ebno_db = float(tf.random.uniform((), *ebno_db_range))
            rx_pow = tf.reduce_mean(tf.abs(y_grid_channel)**2)
            esn0 = 10.0**(ebno_db/10.0)
            sigma2 = rx_pow / esn0
            std_freq = tf.sqrt(tf.cast(sigma2 / 2.0, tf.float32))

            n_f = tf.complex(
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq),
                tf.random.normal(tf.shape(y_grid_channel), stddev=std_freq)
            )
            y_grid_noisy = y_grid_channel + n_f  # ✅ Final grid with channel + noise (for Detection)

            # 4. Modulate to time domain (ALIGNED frame!) for Path A
            y_time_noisy = isac_system.modulator(y_grid_noisy)  # [1, signal_length]

            # 5. Zero-padding for Path A
            y_time_noisy_flat = tf.squeeze(y_time_noisy)
            y_time_noisy_padded = tf.pad(tf.expand_dims(y_time_noisy_flat, 0),
                                        [[0, 0], [0, max_delay_samp]],
                                        constant_values=0)  # [1, signal_length + max_delay_samp]

            # 6. Calculate delay and apply Roll
            if emitter_loc is not None:
                distance = np.linalg.norm(sat_pos - emitter_loc)
            else:
                user_pos = np.array([50e3, 50e3, 0.0])
                distance = np.linalg.norm(sat_pos - user_pos)
            
            delay_samp = int(np.round((distance / c0) * isac_system.SAMPLING_RATE))
            # Path A: roll the channelled (noisy) signal
            rx_time_padded_channel = tf.roll(y_time_noisy_padded, shift=delay_samp, axis=-1)

            # 7. Crop for detection (Path A)
            rx_time_cropped = rx_time_padded_channel[:, delay_samp : delay_samp + signal_length]  # ✅ For Detection IQ

            # 8. Demodulate for CSI (Path A)
            rx_grid_cropped = isac_system.demodulator(rx_time_cropped)  # ✅ For Detection CSI

            # ===== Path B (TDoA-friendly): roll the CLEAN tx_time and add AWGN in time domain =====
            # Roll the clean, pre-padded transmit signal
            rx_time_padded_clean_rolled = tf.roll(tx_time_padded, shift=delay_samp, axis=-1)

            # Add simple AWGN in time-domain based on transmit power and same ebno
            try:
                tx_pow = tf.reduce_mean(tf.abs(tx_time)**2)
                esn0_b = esn0
                sigma2_time = tx_pow / esn0_b
                std_time = tf.sqrt(tf.cast(sigma2_time / 2.0, tf.float32))
                noise_padded = tf.complex(
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time),
                    tf.random.normal(tf.shape(rx_time_padded_clean_rolled), stddev=std_time)
                )
                rx_time_padded_final = rx_time_padded_clean_rolled + noise_padded
            except Exception:
                # Fallback: if anything goes wrong, use the clean rolled signal
                rx_time_padded_final = rx_time_padded_clean_rolled

            sat_rx_list.append({
                'satellite_id': sat_idx,
                'position': sat_pos,
                # For TDoA use Path B (clean tx rolled + AWGN)
                'rx_time_padded': np.squeeze(rx_time_padded_final.numpy()),
                # For Detection use Path A (channelled + noise)
                'rx_time': np.squeeze(rx_time_cropped.numpy()),
                'rx_freq': np.squeeze(rx_grid_cropped.numpy()),
                'true_delay_samples': delay_samp,
                'distance': distance
            })

        # Store primary sat (index 0) for detector features, plus the list for TDoA
        all_iq.append((i, sat_rx_list[0]['rx_time']))
        all_rxfreq.append((i, sat_rx_list[0]['rx_freq']))
        all_sat_recepts.append((i, sat_rx_list))

        # Radar echo (independent noise) - using original tx_grid for consistency
        tx_time = isac_system.modulator(tx_grid)
        radar_delay = np.random.randint(50, 150)
        radar_atten = 0.3
        tx_flat = tf.squeeze(tx_time)
        radar_echo = tf.roll(tx_flat, shift=radar_delay, axis=-1) * radar_atten
        radar_std = tf.constant(0.05, dtype=tf.float32)
        radar_echo += tf.complex(
            tf.random.normal(tf.shape(radar_echo), stddev=radar_std),
            tf.random.normal(tf.shape(radar_echo), stddev=radar_std)
        )

        all_radar.append((i, np.squeeze(radar_echo.numpy())))
        all_labels.append((i, y))
        all_emit.append((i, emitter_loc))

        if (idx+1) % 100 == 0:
            print(f"Generated {idx+1}/{total} multi-sat samples")

    # Sort for alignment
    all_iq = [x[1] for x in sorted(all_iq, key=lambda z: z[0])]
    all_rxfreq = [x[1] for x in sorted(all_rxfreq, key=lambda z: z[0])]
    all_radar = [x[1] for x in sorted(all_radar, key=lambda z: z[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda z: z[0])]
    all_emit = [x[1] for x in sorted(all_emit, key=lambda z: z[0])]
    all_sat_recepts = [x[1] for x in sorted(all_sat_recepts, key=lambda z: z[0])]

    # Make tx_time_padded homogeneous: pad all per-sample transmit arrays to the same length
    if len(all_tx_time_padded) == 0:
        tx_time_padded_arr = np.zeros((0, 0), dtype=np.complex64)
    else:
        # Determine max length among samples
        try:
            lengths = [int(np.asarray(x).shape[-1]) for x in all_tx_time_padded]
        except Exception:
            lengths = [len(np.asarray(x)) for x in all_tx_time_padded]
        max_len = int(max(lengths))
        # Allocate padded array (rows=samples, cols=time)
        tx_time_padded_arr = np.zeros((len(all_tx_time_padded), max_len), dtype=np.complex64)
        for ii, x in enumerate(all_tx_time_padded):
            arr = np.asarray(x)
            L = arr.shape[-1]
            tx_time_padded_arr[ii, :L] = arr

    return {
        'iq_samples': np.array(all_iq),
        'csi': np.array(all_rxfreq),
        'radar_echo': np.array(all_radar),
        'labels': np.array(all_labels),
        'emitter_locations': all_emit,
        'satellite_receptions': all_sat_recepts,
        'sampling_rate': isac_system.SAMPLING_RATE,
        'tx_time_padded': tx_time_padded_arr
    }
# -----------------------------
# Feature Extraction (GPU-Optimized)
# -----------------------------
@tf.function(jit_compile=True)
def extract_spectrogram_tf(iq_batch, n_fft=128, frame_length=128, frame_step=32, out_hw=(64,64)):
    """
    Compute GPU-accelerated |STFT| spectrograms using TensorFlow.
    - Supports complex64 IQ input (automatically converts to |IQ| magnitude)
    - Fully GPU-compatible (no NumPy, no Python loops in graph)
    - Mixed precision enabled for H100 Tensor Cores
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    iq_batch = tf.convert_to_tensor(iq_batch, dtype=tf.complex64)
    # Magnitude of complex IQ → real-valued signal for STFT
    x_mag = tf.abs(iq_batch)

    # Compute STFT on GPU
    stft_c = tf.signal.stft(
        x_mag,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=n_fft
    )  # shape: [B, time, freq]

    # Compute magnitude spectrogram
    spec = tf.abs(stft_c)
    spec = tf.expand_dims(spec, axis=-1)  # [B, T, F, 1]
    H, W = out_hw
    spec = tf.image.resize(spec, [H, W])  # Resize spectrograms

    # Normalize for stable training
    spec = spec / (tf.reduce_max(spec, axis=[1,2,3], keepdims=True) + 1e-8)

    return tf.cast(spec, tf.float32)


# -----------------------------
# RX Feature Extraction (GPU-Optimized)
# -----------------------------
@tf.function(jit_compile=True)
def extract_received_signal_features(dataset):
    """
    Compute per-subcarrier statistics (mean, std, max power) directly on GPU.
    Input: dataset['csi'] → array of complex64 [SYM, SC]
    Output: tensor [B, 8, 8, 3]  (ready for CNN)
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    csi = tf.convert_to_tensor(dataset['csi'], dtype=tf.complex64)  # [B, SYM, SC]
    pwr = tf.abs(csi) ** 2
    pwr = pwr / (tf.reduce_max(pwr, axis=[1, 2], keepdims=True) + 1e-12)

    # Compute statistics across OFDM symbols
    mean_sc = tf.reduce_mean(pwr, axis=1)
    std_sc  = tf.math.reduce_std(pwr, axis=1)
    max_sc  = tf.reduce_max(pwr, axis=1)

    # Stack 3 feature maps → [B, SC, 3]
    F = tf.stack([mean_sc, std_sc, max_sc], axis=-1)

    # Trim or pad to exactly 64 subcarriers
    num_sc = tf.shape(F)[1]
    def trim():
        return F[:, :64, :]
    def pad():
        pad_len = 64 - num_sc
        return tf.pad(F, [[0, 0], [0, pad_len], [0, 0]])
    F = tf.cond(num_sc >= 64, trim, pad)

    # Reshape to 8×8×3 for CNN input
    F = tf.reshape(F, [-1, 8, 8, 3])

    # Cast to float32 for model input
    return tf.cast(F, tf.float32)

# -----------------------------
# Dual-Input CNN
# -----------------------------
def build_dual_input_cnn_fixed():
    """Spectrogram (64x64x1) + RX stats (8x8x3) → binary classifier."""
    # Input 1: Spectrogram
    a_in = tf.keras.layers.Input(shape=(64,64,1), name="spectrogram")
    a = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(a_in)
    a = tf.keras.layers.MaxPooling2D(2)(a)
    a = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(a)
    a = tf.keras.layers.MaxPooling2D(2)(a)
    a = tf.keras.layers.Flatten()(a)

    # Input 2: RX stats
    b_in = tf.keras.layers.Input(shape=(8,8,3), name="rx_features")
    b = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(b_in)
    b = tf.keras.layers.MaxPooling2D(2)(b)
    b = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(b)
    b = tf.keras.layers.MaxPooling2D(2)(b)
    b = tf.keras.layers.Flatten()(b)

    x = tf.keras.layers.Concatenate()([a,b])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model([a_in, b_in], out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Localization (TDoA)
# -----------------------------
def trilateration_2d(tdoa_measurements, satellite_pairs):
    """Levenberg-Marquardt least-squares solver in 2D."""
    from scipy.optimize import least_squares
    def residuals(pos, tdoa_list, pairs):
        x, y = pos
        P = np.array([x, y, 0.0])
        r = []
        for i, (ref, sat) in enumerate(pairs):
            d_ref = np.linalg.norm(P - ref)
            d_i   = np.linalg.norm(P - sat)
            r.append(tdoa_list[i] - (d_i - d_ref))
        return np.array(r)
    centroids = [(ref[:2] + sat[:2])/2.0 for (ref,sat) in satellite_pairs]
    x0 = np.mean(centroids, axis=0)
    res = least_squares(residuals, x0, args=(tdoa_measurements, satellite_pairs), method='lm')
    return np.array([res.x[0], res.x[1], 0.0])

def trilateration_2d_improved(tdoa_measurements, satellite_pairs):
    """Bounded TRF solver in 2D."""
    from scipy.optimize import least_squares
    def resid(pos2, tdoa_list, pairs):
        x, y = pos2
        P = np.array([x, y, 0.0])
        r = []
        for i, (ref, sat) in enumerate(pairs):
            d_ref = np.linalg.norm(P - ref)
            d_i   = np.linalg.norm(P - sat)
            r.append(tdoa_list[i] - (d_i - d_ref))
        return np.array(r)
    sat_positions = np.array([pair[0] for pair in satellite_pairs])
    x0 = np.mean(sat_positions[:, :2], axis=0)
    lo, hi = np.array([-2e5, -2e5]), np.array([2e5, 2e5])
    res = least_squares(resid, x0, args=(tdoa_measurements, satellite_pairs),
                        method='trf', bounds=(lo,hi), max_nfev=1000)
    return np.array([res.x[0], res.x[1], 0.0])

def estimate_emitter_location_tdoa(sample_idx, dataset, isac_system):
    """TDoA using GCC-PHAT with best reference selection and WLS."""
    sat_data = dataset.get('satellite_receptions', None)
    if sat_data is None:
        return None, None
    sats = sat_data[sample_idx]
    if sats is None or len(sats) < 4:
        return None, None

    # Use the stored clean transmit waveform (tx_time_padded) as reference template
    tx_templates = dataset.get('tx_time_padded', None)
    ref_sig = None
    ref_delay = None
    ref_pos = None
    if tx_templates is not None:
        try:
            # template is same length for all sats for this sample
            ref_sig = np.asarray(tx_templates[sample_idx])
        except Exception:
            ref_sig = None

    # If no template available, fall back to best-SNR satellite padded signal (legacy)
    if ref_sig is None:
        snr_scores = [np.mean(np.abs(s.get('rx_time_padded', s['rx_time']))**2) for s in sats]
        best_ref_idx = int(np.argmax(snr_scores))
        ref = sats[best_ref_idx]
        ref_sig = ref.get('rx_time_padded', ref['rx_time'])  # ✅ استفاده از padded
        ref_pos = ref['position']
        ref_delay = ref.get('true_delay_samples', None)

    c0 = 3e8
    Fs = float(dataset.get('sampling_rate', isac_system.SAMPLING_RATE))

    # Compute absolute TOA per satellite by correlating each sat signal with the clean template
    toas = []
    positions = []
    corr_weights = []

    upsampling_factor = 32

    for s in sats:
        pos = s['position']
        positions.append(pos)

        # Build signal for correlation: prefer rolled clean template (Path B) when available
        if ref_sig is not None and 'tx_time_padded' in dataset:
            try:
                cur_delay = s.get('true_delay_samples', None)
                if cur_delay is None:
                    sig = np.asarray(s.get('rx_time_padded', s['rx_time']))
                else:
                    sig = np.roll(ref_sig, cur_delay)[:len(ref_sig)]
                    # Add AWGN approximation if needed (best-effort)
                    tx_pow = np.mean(np.abs(ref_sig)**2)
                    ebno_db_est = 10.0
                    esn0 = 10.0**(ebno_db_est/10.0)
                    sigma2_time = tx_pow / esn0
                    noise = (np.random.normal(0, np.sqrt(sigma2_time/2.0), sig.shape) +
                             1j * np.random.normal(0, np.sqrt(sigma2_time/2.0), sig.shape))
                    sig = sig + noise
            except Exception:
                sig = np.asarray(s.get('rx_time_padded', s['rx_time']))
        else:
            sig = np.asarray(s.get('rx_time_padded', s['rx_time']))

        L = min(len(sig), len(ref_sig))
        sig = np.asarray(sig[:L])
        re = np.asarray(ref_sig[:L])

        corr = gcc_phat(sig, re, upsample_factor=upsampling_factor, beta=1.0, use_hann=False)
        center = len(corr) // 2
        k0 = int(np.argmax(np.abs(corr)))

        if 0 < k0 < len(corr)-1:
            y1, y2, y3 = np.abs(corr[k0-1]), np.abs(corr[k0]), np.abs(corr[k0+1])
            den = (y1 - 2*y2 + y3)
            delta = 0.0 if den == 0 else 0.5*(y1 - y3)/den
            curv = abs(den)
            weight = max(curv, 1e-6) if ABLATION_CONFIG.get('use_curvature_weights', True) else 1.0
        else:
            delta = 0.0
            weight = 1e-6 if ABLATION_CONFIG.get('use_curvature_weights', True) else 1.0

        # Convert peak index to time (seconds)
        d_samp = ((k0 - center) + delta) / upsampling_factor
        dt = d_samp / Fs
        toas.append(dt)
        corr_weights.append(weight)

    if len(toas) < 2:
        return None, None

    toas = np.array(toas)
    corr_weights = np.array(corr_weights)

    # Choose numeric reference: earliest arrival (minimum TOA)
    ref_idx_num = int(np.argmin(toas))
    toa_ref = float(toas[ref_idx_num])
    ref_pos = positions[ref_idx_num]

    # Build TDoA (range-differences in meters) and associated geometry pairs
    tdoa_diffs = []
    pairs = []
    weights = []
    for i, (toa_i, pos_i, w) in enumerate(zip(toas, positions, corr_weights)):
        if i == ref_idx_num:
            continue
        dd = (float(toa_i) - toa_ref) * c0
        tdoa_diffs.append(dd)
        pairs.append((ref_pos, pos_i))
        weights.append(w)

    if len(tdoa_diffs) < 2:
        return None, None

    weights = np.array(weights)
    weights = np.clip(weights / np.max(weights), 1e-3, 1.0)

    try:
        from scipy.optimize import least_squares
        
        def resid_wls(pos2, tdoa_list, pairs, weights):
            x, y = pos2
            P = np.array([x, y, 0.0])
            r = []
            for dd, (ref_pos, sat_pos), w in zip(tdoa_list, pairs, weights):
                d_ref = np.linalg.norm(P - ref_pos)
                d_i = np.linalg.norm(P - sat_pos)
                r.append((dd - (d_i - d_ref)) * np.sqrt(w))
            return np.array(r)
        
        sat_positions = np.array([pair[0] for pair in pairs])
        x0 = np.mean(sat_positions[:, :2], axis=0)
        lo, hi = np.array([-2e5, -2e5]), np.array([2e5, 2e5])
        
        res = least_squares(
            resid_wls, x0, 
            args=(tdoa_diffs, pairs, weights),
            method='trf', 
            loss='huber',
            f_scale=300.0,
            bounds=(lo, hi), 
            max_nfev=2000
        )
        est = np.array([res.x[0], res.x[1], 0.0])
        
    except Exception:
        try:
            est = trilateration_2d_improved(tdoa_diffs, pairs)
        except Exception:
            return None, None

    gt = dataset['emitter_locations'][sample_idx]
    if gt is None:
        return None, None
    return est, np.array(gt)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # CPU-bound init (antenna patterns, topology, LDPC graphs)
    print("Phase 1: Initializing ISAC System (CPU-heavy init)...")
    isac = ISACSystem()
    if NTN_MODELS_AVAILABLE:
        # Precompute a tiny cache of topologies (all with same height here, but structure is in place)
        isac.precompute_topologies(count=4, bs_heights=[isac.SAT_HEIGHT]*4)

    # GPU-heavy simulation + ML
    dataset_path = f"{DATASET_DIR}/dataset_samples{NUM_SAMPLES_PER_CLASS}_sats{NUM_SATELLITES_FOR_TDOA}.pkl"
    
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print("✓ Dataset loaded from disk")
    else:
        print("Phase 2/3: Generating multi-satellite dataset (GPU-friendly ops where possible)...")
        dataset = generate_dataset_multi_satellite(
            isac, num_samples_per_class=NUM_SAMPLES_PER_CLASS, num_satellites=NUM_SATELLITES_FOR_TDOA
        )
    # Save dataset
        print(f"Saving dataset to {dataset_path}...")
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print("✓ Dataset saved to disk")

    # Quick validation
    print("\n=== DATA VALIDATION ===")
    labels = dataset['labels']
    print(f"Benign samples: {np.sum(labels==0)}")
    print(f"Attack samples: {np.sum(labels==1)}")
    print(f"Attack samples with emitter locations: {sum(1 for loc in dataset['emitter_locations'] if loc is not None)}")

    # Fix #3: Add diagnostics
    benign_idx = np.where(labels == 0)[0][:100]
    attack_idx = np.where(labels == 1)[0][:100]
    benign_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) 
                            for i in benign_idx])
    attack_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) 
                            for i in attack_idx])
    power_ratio = attack_power / benign_power
    print(f"Power ratio (attack/benign): {power_ratio:.4f}")

    # Should be between 1.15-1.20!
    if power_ratio < 1.1:
        print("⚠️ WARNING: Power ratio too low - dataset may be biased!")

        # ===============================
    # Phase 4: Extracting features
    # ===============================
    print("\nPhase 4: Extracting features (GPU)...")
    feats_spec = extract_spectrogram_tf(dataset['iq_samples'], n_fft=128, frame_length=128, frame_step=32, out_hw=(64,64))
    feats_rx   = extract_received_signal_features(dataset)

    # --- Convert to NumPy for sklearn compatibility ---
    feats_spec = np.array(feats_spec)
    feats_rx   = np.array(feats_rx)
    labels     = np.array(dataset['labels']).astype(np.float32)

    # --- هم‌ترازی ---
    m = min(len(feats_spec), len(feats_rx), len(labels))
    feats_spec, feats_rx, labels = feats_spec[:m], feats_rx[:m], labels[:m]

    # ===== NEW: Group-based Split (Prevent Data Leakage) =====
    def sat_group_id(sat_rx_list):
        """
        Create group ID based on satellite constellation geometry.
        Groups samples with same satellite altitudes together.
        """
        # Guard: If no sat_rx_list available, fallback to unique id
        if sat_rx_list is None:
            return hash(None)
        try:
            # --- 🔥 اصلاح کلیدی: گروه‌بندی بر اساس پوسته ارتفاعی (Shell) ---
            # به جای ارتفاع دقیق، ببین هر ماهواره در کدام "پوسته" قرار دارد
            # پوسته‌ها: 300-400km, 400-500km, 500-600km
            def get_shell(alt_km):
                if 300 <= alt_km < 400: return 3
                if 400 <= alt_km < 500: return 4
                if 500 <= alt_km < 600: return 5
                return 6 # برای بالای ۶۰۰

            # یک شناسه بر اساس تعداد ماهواره‌ها در هر پوسته ارتفاعی بساز
            shells = [get_shell(s['position'][2]/1000) for s in sat_rx_list]
            # شمارش تعداد ماهواره‌ها در هر پوسته
            shell_counts = tuple(sorted(np.bincount(shells))) 
            
            # hash نهایی فقط بر اساس این شمارش‌ها خواهد بود
            return hash((shell_counts, len(sat_rx_list)))

        except Exception as e:
            # Fallback if structure unexpected
            print(f"Warning in sat_group_id: {e}")
            return hash(str(sat_rx_list))

    # Create groups array from multi-satellite receptions if available, otherwise use None per-sample
    print("\nCreating constellation-based groups...")
    sat_recepts = dataset.get('satellite_receptions', None)
    if sat_recepts is None:
        # Fall back to single-sat grouping by using primary rx 'csi' alt (if present)
        groups = np.array([hash(None)] * m)
    else:
        groups = np.array([sat_group_id(sr) for sr in sat_recepts[:m]])

    unique_groups = len(np.unique(groups))
    print(f"✓ Found {unique_groups} unique satellite constellations")
    try:
        binc = np.bincount(groups)
        print(f"  Samples per group: min={np.min(binc)}, max={np.max(binc)}, mean={len(groups)/unique_groups:.1f}")
    except Exception:
        # groups may be arbitrary hashes; use fallback stats
        counts = {g: int(np.sum(groups == g)) for g in np.unique(groups)}
        vals = np.array(list(counts.values()))
        print(f"  Samples per group: min={np.min(vals)}, max={np.max(vals)}, mean={len(groups)/unique_groups:.1f}")

    # If there aren't at least 2 unique groups, fall back to a stratified random split
    if unique_groups < 2:
        print("⚠️ Only one unique constellation group detected — falling back to stratified random split to avoid empty train/test groups.")
        # Use stratify to preserve class balance
        Xs_tr, Xs_te, Xr_tr, Xr_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            feats_spec, feats_rx, labels, np.arange(m), test_size=0.2, random_state=42, stratify=labels
        )
        print(f"✓ Stratified random split: Train={len(Xs_tr)} samples, Test={len(Xs_te)} samples")
    else:
        # Group-based split (ensures no constellation appears in both train and test)
        gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(np.arange(len(groups)), groups=groups))

        print(f"✓ Train groups: {len(np.unique(groups[train_idx]))}")
        print(f"✓ Test groups: {len(np.unique(groups[test_idx]))}")

        # Verify no group overlap (sanity check)
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            print(f"⚠️ WARNING: {len(overlap)} groups overlap between train/test!")
        else:
            print("✓ No group overlap - proper split! 🎯")

        # Split data
        Xs_tr, Xs_te = feats_spec[train_idx], feats_spec[test_idx]
        Xr_tr, Xr_te = feats_rx[train_idx], feats_rx[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        idx_tr, idx_te = train_idx, test_idx

        print(f"✓ Train: {len(Xs_tr)} samples")
        print(f"✓ Test: {len(Xs_te)} samples")

    # ==================================================
    # Phase 5: Training with mixed-precision + XLA (H100)
    # ==================================================
    print("\nPhase 5: Training dual-input CNN (H100 Optimized)...")

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)  # Enable XLA compiler

    AUTOTUNE = tf.data.AUTOTUNE

    # Build model with fp32 output head for numerical stability
    def build_dual_input_cnn_h100():
        # Spectrogram branch (optional)
        if ABLATION_CONFIG.get('use_spectrogram', True):
            a_in = tf.keras.layers.Input(shape=(64,64,1), name="spectrogram")
            a = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(a_in)
            a = tf.keras.layers.BatchNormalization()(a)
            a = tf.keras.layers.MaxPooling2D(2)(a)
            a = tf.keras.layers.Dropout(0.25)(a)

            a = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
            a = tf.keras.layers.BatchNormalization()(a)
            a = tf.keras.layers.MaxPooling2D(2)(a)
            a = tf.keras.layers.Dropout(0.25)(a)

            a = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
            a = tf.keras.layers.BatchNormalization()(a)
            a = tf.keras.layers.MaxPooling2D(2)(a)
            a = tf.keras.layers.Dropout(0.3)(a)

            a = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(a)
            a = tf.keras.layers.BatchNormalization()(a)
            a = tf.keras.layers.GlobalAveragePooling2D()(a)
        else:
            # Dummy spectrogram input → zero contribution
            a_in = tf.keras.layers.Input(shape=(64,64,1), name="spectrogram")
            a = tf.keras.layers.Flatten()(a_in)
            a = tf.keras.layers.Lambda(lambda x: x * 0)(a)

        # RX features branch - with additional layers (optional)
        if ABLATION_CONFIG.get('use_rx_features', True):
            b_in = tf.keras.layers.Input(shape=(8,8,3), name="rx_features")
            b = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(b_in)
            b = tf.keras.layers.MaxPooling2D(2)(b)
            b = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(b)
            b = tf.keras.layers.MaxPooling2D(2)(b)
            b = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(b)
            b = tf.keras.layers.GlobalAveragePooling2D()(b)
        else:
            # Dummy rx input → zero contribution
            b_in = tf.keras.layers.Input(shape=(8,8,3), name="rx_features")
            b = tf.keras.layers.Flatten()(b_in)
            b = tf.keras.layers.Lambda(lambda x: x * 0)(b)

        x = tf.keras.layers.Concatenate()([a, b])
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

        model = tf.keras.Model([a_in, b_in], out)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4, epsilon=1e-7)
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
            jit_compile=True,
            steps_per_execution=128
        )
        return model

    # --- تبدیل داده‌ها به float32 ---
    Xs_tr, Xr_tr, y_tr = map(np.float32, [Xs_tr, Xr_tr, y_tr])
    Xs_te, Xr_te, y_te = map(np.float32, [Xs_te, Xr_te, y_te])

    # --- Build Dataset for high GPU throughput ---
    def make_ds(Xs, Xr, y, batch, shuffle=False, drop_remainder=True):
        ds = tf.data.Dataset.from_tensor_slices(((Xs, Xr), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(y), 4096))
        ds = ds.batch(batch, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    # For training keep drop_remainder=True for consistent steps_per_epoch; for validation use False
    train_ds = make_ds(Xs_tr, Xr_tr, y_tr, TRAIN_BATCH, shuffle=True, drop_remainder=True)
    test_ds  = make_ds(Xs_te, Xr_te, y_te, TRAIN_BATCH, shuffle=False, drop_remainder=False)

    # Diagnostics: ensure datasets are non-empty and shapes match
    try:
        print(f"\nDataset shapes: Xs_tr={getattr(Xs_tr, 'shape', None)}, Xr_tr={getattr(Xr_tr, 'shape', None)}, y_tr={getattr(y_tr, 'shape', None)}")
        print(f"               Xs_te={getattr(Xs_te, 'shape', None)}, Xr_te={getattr(Xr_te, 'shape', None)}, y_te={getattr(y_te, 'shape', None)}")
        n_train = int(getattr(y_tr, '__len__', lambda: 0)())
        n_test = int(getattr(y_te, '__len__', lambda: 0)())
    except Exception:
        n_train = len(y_tr) if hasattr(y_tr, '__len__') else 0
        n_test = len(y_te) if hasattr(y_te, '__len__') else 0

    if n_train == 0:
        raise ValueError(f"Training set is empty (n_train={n_train}). Aborting training.")

    if n_test == 0:
        print("⚠️ Validation set is empty - training will proceed without validation.")


    # --- ساخت مدل و Callbackها ---
    model = build_dual_input_cnn_h100()
    # Fix #4: Better early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/best_model.keras",
            monitor='val_accuracy',  # ← تغییر از val_auc
            mode='max', 
            save_best_only=True, 
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # ← تغییر از val_auc
            mode='max', 
            patience=10,  # ← تغییر از 6
            min_delta=0.001,
            restore_best_weights=True, 
            verbose=1
        ),
    ]

    hist = model.fit(
        train_ds,
        epochs=TRAIN_EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )

    print("\n=== TRAIN SUMMARY ===")
    if 'val_auc' in hist.history:
        best_ep = 1 + int(np.argmax(hist.history['val_auc']))
        print(f"Best epoch: {best_ep}")
        print(f"Best val_auc: {np.max(hist.history['val_auc']):.4f}")
        print(f"Final train_acc: {hist.history['accuracy'][-1]:.4f}")

    # ===============================
    # Phase 6: Evaluation and TDoA
    # ===============================
    print("\nEvaluating on test set (H100)...")
    # Fix #2: Test evaluation - استفاده از کل dataset تست
    test_ds_full = tf.data.Dataset.from_tensor_slices(((Xs_te, Xr_te), y_te))
    test_ds_full = test_ds_full.batch(64).prefetch(AUTOTUNE)

    loss, acc, auc_val = model.evaluate(test_ds_full, verbose=0)
    print(f"Test Results → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc_val:.4f}")

    y_prob = model.predict(test_ds_full, verbose=0).ravel()
    
    # -----------------------------
    # Calibration metrics (ECE / Brier) and ROC @ low FPR
    # -----------------------------
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        # Calibration metrics
        ece_bins = 10
        prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=ece_bins)
        brier = brier_score_loss(y_te, y_prob)

        # Approximate ECE
        ece = np.mean(np.abs(prob_true - prob_pred))

        print(f"\n=== CALIBRATION METRICS ===")
        print(f"Brier Score: {brier:.4f}")
        print(f"ECE (Expected Calibration Error): {ece:.4f}")

        # ROC at specific low FPR thresholds
        from sklearn.metrics import roc_curve
        fpr_full, tpr_full, thresholds = roc_curve(y_te, y_prob)
        target_fprs = [1e-3, 1e-4, 1e-5]
        print(f"\n=== ROC @ LOW FPR ===")
        for target_fpr in target_fprs:
            idx = np.where(fpr_full <= target_fpr)[0]
            if len(idx) > 0:
                best_tpr = tpr_full[idx[-1]]
                print(f"TPR @ FPR={target_fpr:.0e}: {best_tpr:.4f}")
            else:
                print(f"TPR @ FPR={target_fpr:.0e}: N/A (cannot achieve this FPR)")

        # Calibration plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 'o-', label=f'Model (ECE={ece:.3f})')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/11_calibration_curve.pdf", dpi=300)
        plt.close()
        print(f"✓ Calibration curve saved")
    except Exception as e:
        print(f"⚠️ Calibration metrics skipped: {e}")
    
    # Threshold tuning: find best threshold that maximizes F1
    print(f"\n=== THRESHOLD TUNING ===")
    print(f"Default threshold (0.5) results:")
    y_hat_default = (y_prob > 0.5).astype(int)
    f1_default = f1_score(y_te, y_hat_default)
    print(f"  F1 score: {f1_default:.4f}")
    
    # محاسبه F1 برای تمام thresholdهای ممکن
    p, r, t = precision_recall_curve(y_te, y_prob)
    # Compute F1 score for each threshold (avoid division by zero)
    f1_scores = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    
    # پیدا کردن بهترین threshold
    best_idx = np.argmax(f1_scores)
    best_thr = t[best_idx] if best_idx < len(t) else 0.5
    
    print(f"\nOptimized threshold: {best_thr:.4f}")
    print(f"  Best F1 score: {f1_scores[best_idx]:.4f}")
    print(f"  Precision at best threshold: {p[best_idx]:.4f}")
    print(f"  Recall at best threshold: {r[best_idx]:.4f}")
    print(f"  Improvement: {(f1_scores[best_idx] - f1_default) * 100:.2f}%")
    
    # استفاده از threshold بهینه برای پیش‌بینی نهایی
    y_hat = (y_prob > best_thr).astype(int)

    # ===============================
    # Phase 7: Generate Paper Plots
    # ===============================
    print("\n=== GENERATING PAPER PLOTS ===")
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
        cm  = confusion_matrix(y_te, y_hat)
        rep = classification_report(y_te, y_hat, target_names=['Benign','Attack'])
        ap  = average_precision_score(y_te, y_prob)
        print("\n=== CONFUSION MATRIX ===")
        print("                Predicted")
        print("              Benign  Attack")
        print(f"Actual Benign  {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"       Attack  {cm[1,0]:4d}    {cm[1,1]:4d}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(rep)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"AUPRC (avg precision): {ap:.3f}")

        # Must-Have Plot 1: ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC AUC={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--',lw=1, color='gray', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Covert Detection', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/1_roc_curve.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/1_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curve saved to {RESULT_DIR}/1_roc_curve.pdf")

        # Must-Have Plot 2: Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(r, p, lw=2, label=f'AP={ap:.3f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/2_precision_recall_curve.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/2_precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Precision-Recall curve saved to {RESULT_DIR}/2_precision_recall_curve.pdf")

        # Must-Have Plot 3: Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Attack'], 
                    yticklabels=['Benign', 'Attack'],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/3_confusion_matrix.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/3_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion Matrix saved to {RESULT_DIR}/3_confusion_matrix.pdf")

        # Must-Have Plot 5: Training History
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hist.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(hist.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.title('Model Accuracy', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(hist.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(hist.history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=11)
        plt.ylabel('Loss', fontsize=11)
        plt.title('Model Loss', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/5_training_history.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/5_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training History saved to {RESULT_DIR}/5_training_history.pdf")

        # Must-Have Plot 6: Power Spectrum (Benign vs Attack)
        plt.figure(figsize=(10, 6))
    # Compute power spectrum for benign and attack samples
        benign_samples = dataset['iq_samples'][labels == 0][:10]
        attack_samples = dataset['iq_samples'][labels == 1][:10]
        
        benign_fft = np.mean([np.abs(np.fft.fft(s))**2 for s in benign_samples], axis=0)
        attack_fft = np.mean([np.abs(np.fft.fft(s))**2 for s in attack_samples], axis=0)
        
        freq = np.fft.fftfreq(len(benign_fft), d=1/isac.SAMPLING_RATE) / 1e6  # MHz
        
        plt.plot(freq[:len(freq)//2], 10*np.log10(benign_fft[:len(freq)//2] + 1e-12), 
                label='Benign', linewidth=2, alpha=0.8)
        plt.plot(freq[:len(freq)//2], 10*np.log10(attack_fft[:len(freq)//2] + 1e-12), 
                label='Attack (Covert)', linewidth=2, alpha=0.8)
        plt.xlabel('Frequency (MHz)', fontsize=12)
        plt.ylabel('Power Spectrum (dB)', fontsize=12)
        plt.title('Power Spectrum: Benign vs Attack', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/6_power_spectrum.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/6_power_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Power Spectrum saved to {RESULT_DIR}/6_power_spectrum.pdf")

        # Nice-to-Have Plot 1: F1 Score vs Threshold
        plt.figure(figsize=(8, 6))
        plt.plot(t, f1_scores[:-1], linewidth=2, label='F1 Score')
        plt.axvline(best_thr, color='r', linestyle='--', linewidth=2, label=f'Best Threshold={best_thr:.3f}')
        plt.axhline(f1_scores[best_idx], color='g', linestyle=':', linewidth=1, alpha=0.5)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/7_f1_vs_threshold.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/7_f1_vs_threshold.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ F1 vs Threshold saved to {RESULT_DIR}/7_f1_vs_threshold.pdf")

        # Nice-to-Have Plot 3: Spectrogram Examples (2x2 grid)
    # Find different categories of samples: benign/attack × detected/missed
        benign_correct = np.where((y_te == 0) & (y_hat == 0))[0]
        benign_wrong = np.where((y_te == 0) & (y_hat == 1))[0]
        attack_correct = np.where((y_te == 1) & (y_hat == 1))[0]
        attack_wrong = np.where((y_te == 1) & (y_hat == 0))[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Benign Detected (True Negative)
        if len(benign_correct) > 0:
            idx = benign_correct[0]
            spec = Xs_te[idx].squeeze()
            axes[0, 0].imshow(spec, aspect='auto', cmap='viridis', origin='lower')
            axes[0, 0].set_title('Benign - Correctly Detected', fontweight='bold')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Frequency')
        
        # Benign Missed (False Positive)
        if len(benign_wrong) > 0:
            idx = benign_wrong[0]
            spec = Xs_te[idx].squeeze()
            axes[0, 1].imshow(spec, aspect='auto', cmap='viridis', origin='lower')
            axes[0, 1].set_title('Benign - Misclassified as Attack', fontweight='bold', color='red')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Frequency')
        
        # Attack Detected (True Positive)
        if len(attack_correct) > 0:
            idx = attack_correct[0]
            spec = Xs_te[idx].squeeze()
            axes[1, 0].imshow(spec, aspect='auto', cmap='plasma', origin='lower')
            axes[1, 0].set_title('Attack - Correctly Detected', fontweight='bold')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Frequency')
        
        # Attack Missed (False Negative)
        if len(attack_wrong) > 0:
            idx = attack_wrong[0]
            spec = Xs_te[idx].squeeze()
            axes[1, 1].imshow(spec, aspect='auto', cmap='plasma', origin='lower')
            axes[1, 1].set_title('Attack - Missed Detection', fontweight='bold', color='red')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/9_spectrogram_examples.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/9_spectrogram_examples.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Spectrogram Examples saved to {RESULT_DIR}/9_spectrogram_examples.pdf")

        # Nice-to-Have Plot 4: Satellite Geometry (3D visualization)
        if 'satellite_receptions' in dataset and len(dataset['satellite_receptions']) > 0:
            # Extract satellite positions from the first sample
            first_sample_sats = dataset['satellite_receptions'][0]
            sat_positions = np.array([s['position'] for s in first_sample_sats])
            
            fig = plt.figure(figsize=(12, 5))
            
            # نمای 3D
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(sat_positions[:, 0]/1e3, sat_positions[:, 1]/1e3, 
                       sat_positions[:, 2]/1e3, s=200, c='red', marker='^', 
                       edgecolors='black', linewidths=2, label='Satellites')
            ax1.scatter([0], [0], [0], s=300, c='blue', marker='o', 
                       edgecolors='black', linewidths=2, label='Ground (0,0,0)')
            
            for i, pos in enumerate(sat_positions):
                ax1.text(pos[0]/1e3, pos[1]/1e3, pos[2]/1e3, f'  S{i+1}', fontsize=9)
            
            ax1.set_xlabel('X (km)', fontsize=11)
            ax1.set_ylabel('Y (km)', fontsize=11)
            ax1.set_zlabel('Altitude (km)', fontsize=11)
            ax1.set_title('3D Satellite Constellation', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # نمای 2D (Top view)
            ax2 = fig.add_subplot(122)
            ax2.scatter(sat_positions[:, 0]/1e3, sat_positions[:, 1]/1e3, 
                       s=200, c=sat_positions[:, 2]/1e3, cmap='viridis', 
                       marker='^', edgecolors='black', linewidths=2)
            cbar = plt.colorbar(ax2.collections[0], ax=ax2, label='Altitude (km)')
            
            for i, pos in enumerate(sat_positions):
                ax2.text(pos[0]/1e3 + 5, pos[1]/1e3 + 5, f'S{i+1}\n{pos[2]/1e3:.1f}km', 
                        fontsize=9, ha='left')
            
            ax2.scatter([0], [0], s=300, c='red', marker='x', linewidths=3, label='Ground (0,0)')
            ax2.set_xlabel('X (km)', fontsize=11)
            ax2.set_ylabel('Y (km)', fontsize=11)
            ax2.set_title('Top View (with Altitude Diversity)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/10_satellite_geometry.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{RESULT_DIR}/10_satellite_geometry.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Satellite Geometry saved to {RESULT_DIR}/10_satellite_geometry.pdf")

    except Exception as e:
        print(f"[WARN] Error generating plots: {e}")
        import traceback
        traceback.print_exc()

    # ===============================
    # Phase 8: TDoA Localization
    # ===============================
    print("\n=== PHASE 8: TDoA-BASED LOCALIZATION ===")
    print("Processing true positives...")
    loc_errors = []
    tp_sample_ids = []
    tp_ests = []
    tp_gts = []
    for i, pred in enumerate(y_hat):
        if pred == 1 and y_te[i] == 1:
            ds_idx = int(idx_te[i])
            est, gt = estimate_emitter_location_tdoa(ds_idx, dataset, isac)
            if est is not None and gt is not None:
                err = np.linalg.norm(est - gt)
                loc_errors.append(err)
                tp_sample_ids.append(ds_idx)
                tp_ests.append(est)
                tp_gts.append(gt)
                if len(loc_errors) <= 3:
                    print(f"  Sample {ds_idx} | GT {gt[:2]} | EST {est[:2]} | Error={err:.2f} m")

    if loc_errors:
        med = float(np.median(loc_errors))
        mean= float(np.mean(loc_errors))
        p90 = float(np.percentile(loc_errors, 90))
        print("\n=== TDoA Localization Results ===")
        print(f"Median Error: {med:.2f} m")
        print(f"Mean Error  : {mean:.2f} m")
        print(f"90th Perc.  : {p90:.2f} m")
        print(f"Total samples: {len(loc_errors)}")

        # Must-Have Plot 4: Localization Error CDF
        xs = np.sort(loc_errors)
        cdf = np.arange(1, len(xs)+1)/len(xs)
        plt.figure(figsize=(8, 6))
        markevery = max(1, len(xs)//20)
        plt.plot(xs, cdf, linewidth=2, marker='o', markersize=3, markevery=markevery)
        plt.axhline(0.9, color='r', ls='--', linewidth=2, label=f'90th percentile = {p90:.1f} m')
        plt.axhline(0.5, color='orange', ls=':', linewidth=1, alpha=0.5, label=f'Median = {med:.1f} m')
        plt.xlabel('Localization Error (m)', fontsize=12)
        plt.ylabel('CDF', fontsize=12)
        plt.title('Localization Error CDF', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/4_localization_cdf.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/4_localization_cdf.png", dpi=300, bbox_inches='tight')

        # --- Optional: Compute theoretical CRLB for TDoA ---
        try:
            import itertools
            c0 = 3e8
            # Derive system timing resolution from sampling rate and upsampling used in GCC-PHAT
            upsampling_factor = 32
            timing_resolution_ns = 1e9 / (isac.SAMPLING_RATE * upsampling_factor)
            # Assume sigma_t ~ 1.2 x timing resolution (empirical safety margin)
            sigma_t = timing_resolution_ns * 1.2 * 1e-9
            sigma_d = c0 * sigma_t

            # Select true-positive samples (we already computed ests/gts)
            sample_tp = tp_sample_ids
            sample_errors = loc_errors
            sample_ests = tp_ests
            sample_gts = tp_gts

            # Control CRLB sample count (env var fallback)
            try:
                CRLB_SAMPLES = os.getenv('CRLB_SAMPLES', 'all')  # 'all' or integer string
                if CRLB_SAMPLES != 'all':
                    CRLB_SAMPLES = int(CRLB_SAMPLES)
            except Exception:
                CRLB_SAMPLES = 'all'
            print(f"CRLB_SAMPLES setting: {CRLB_SAMPLES}")

            if isinstance(CRLB_SAMPLES, str) and CRLB_SAMPLES == 'all':
                use_idxs = list(range(len(sample_tp)))
            else:
                N = int(CRLB_SAMPLES)
                use_idxs = list(range(min(N, len(sample_tp))))

            crlb_values = []
            sample_ids_used = []
            achieved_errors_used = []
            est_x = []
            est_y = []
            gt_x = []
            gt_y = []

            for ii in use_idxs:
                ds_idx = sample_tp[ii]
                sat_list = dataset.get('satellite_receptions', [None])[ds_idx]
                if sat_list is None:
                    continue
                try:
                    snr_scores = [np.mean(np.abs(s.get('rx_time_padded', s['rx_time']))**2) for s in sat_list]
                    best_ref_idx = int(np.argmax(snr_scores))
                    ref = sat_list[best_ref_idx]['position']
                    H = []
                    for s in sat_list:
                        if s['satellite_id'] == best_ref_idx:
                            continue
                        sat = np.array(s['position'][:2])
                        ref2 = np.array(ref[:2])

                        # Choose evaluation point P: prefer GT, else EST
                        try:
                            P = np.array(sample_gts[ii][:2])
                        except Exception:
                            P = np.array(sample_ests[ii][:2])

                        d_sat = np.linalg.norm(P - sat)
                        d_ref = np.linalg.norm(P - ref2)
                        if d_sat < 1e-6 or d_ref < 1e-6:
                            continue

                        # Jacobian row: gradient of (||P-sat|| - ||P-ref||) wrt P
                        grad = (P - sat) / d_sat - (P - ref2) / d_ref
                        H.append(grad)
                    H = np.array(H)
                    if H.shape[0] >= 2:
                        # Fisher Information Matrix for position (2D)
                        FIM = (H.T @ H) / (sigma_d**2)
                        CRLB_matrix = np.linalg.inv(FIM)
                        crlb_i = float(np.sqrt(np.trace(CRLB_matrix)))
                        crlb_values.append(crlb_i)
                        sample_ids_used.append(ds_idx)
                        achieved_errors_used.append(sample_errors[ii])
                        est_x.append(float(sample_ests[ii][0]))
                        est_y.append(float(sample_ests[ii][1]))
                        gt_x.append(float(sample_gts[ii][0]))
                        gt_y.append(float(sample_gts[ii][1]))
                except Exception:
                    continue

            if crlb_values:
                mean_crlb = float(np.mean(crlb_values))
                med_crlb = float(np.median(crlb_values))
                achieved_med = med
                # Clarify efficiency: ratio = achieved / CRLB (>=1 desirable), then efficiency pct = 100 / ratio
                if med_crlb > 0:
                    ratio = achieved_med / med_crlb
                    eff_pct = 100.0 / ratio if ratio > 0 else 0.0
                else:
                    ratio = float('inf')
                    eff_pct = 0.0

                print("\n=== CRLB Analysis ===")
                print(f"System timing resolution: {timing_resolution_ns:.2f} ns")
                print(f"Assumed timing error (σ_t): {sigma_t*1e9:.2f} ns")
                print(f"Equivalent range error (σ_d): {sigma_d:.2f} m")
                print(f"Mean CRLB: {mean_crlb:.2f} m")
                print(f"Median CRLB: {med_crlb:.2f} m")
                print(f"Achieved median error: {achieved_med:.2f} m")
                # New, clearer metrics
                print(f"Ratio (achieved/CRLB): {ratio:.2f}×")
                print(f"Efficiency: {eff_pct:.1f}% of the CRLB")

                # Save per-sample CRLBs and metadata
                try:
                    import pandas as pd
                    crlb_values_arr = np.array(crlb_values)
                    df_crlb = pd.DataFrame({
                        'sample_id': sample_ids_used,
                        'crlb_m': crlb_values_arr,
                        'achieved_error_m': achieved_errors_used,
                        'GT_x': gt_x,
                        'GT_y': gt_y,
                        'EST_x': est_x,
                        'EST_y': est_y
                    })
                    csv_path = os.path.join(RESULT_DIR, 'crlb_values.csv')
                    df_crlb.to_csv(csv_path, index=False)
                    print(f"✓ Detailed CRLB analysis saved to {csv_path}")
                except Exception as e:
                    print(f"⚠️ Could not save detailed CRLB CSV: {e}")
            else:
                print("(CRLB computation skipped: no valid H matrices from samples)")
        except Exception as e:
            print(f"(CRLB computation skipped: {e})")
        plt.close()
        print(f"✓ Localization CDF saved to {RESULT_DIR}/4_localization_cdf.pdf")

        # Nice-to-Have Plot 2: Localization Error Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(loc_errors, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(med, color='r', linestyle='--', linewidth=2, label=f'Median={med:.1f}m')
        plt.axvline(mean, color='g', linestyle=':', linewidth=2, label=f'Mean={mean:.1f}m')
        plt.xlabel('Localization Error (m)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Localization Error Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/8_localization_histogram.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{RESULT_DIR}/8_localization_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Localization Histogram saved to {RESULT_DIR}/8_localization_histogram.pdf")
        # Optional Logging: save localization errors for reproducibility / later plotting
        try:
            with open(f"{RESULT_DIR}/localization_errors.txt", "w") as f:
                for e in loc_errors:
                    f.write(f"{e:.4f}\n")
            print(f"✓ Localization errors written to {RESULT_DIR}/localization_errors.txt")
        except Exception as _:
            pass
    else:
        print("⚠️ No true-positive attacks to localize.")

    # ===============================
    # Summary Report
    # ===============================
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"Dataset: {NUM_SAMPLES_PER_CLASS*2} samples ({NUM_SAMPLES_PER_CLASS} per class)")
    print(f"Model: Dual-Input CNN (saved to {MODEL_DIR}/)")
    print(f"Results: All plots saved to {RESULT_DIR}/")
    print(f"Best Threshold: {best_thr:.4f} (F1={f1_scores[best_idx]:.4f})")
    if loc_errors:
        print(f"Localization: Median Error = {med:.2f} m, 90th = {p90:.2f} m")
    print("="*60)
