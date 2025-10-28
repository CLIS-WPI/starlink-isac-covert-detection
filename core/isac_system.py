# ======================================
# ðŸ“„ core/isac_system.py
# Purpose: ISAC System with optimized topology caching
# OPTIMIZED: Pre-generates 1000+ topologies for reuse
# ======================================

import numpy as np
import os
import tensorflow as tf
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, OFDMDemodulator, OFDMModulator, ResourceGridMapper
from sionna.phy.channel import RayleighBlockFading
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.channel import time_lag_discrete_time_channel
from config.settings import *

# Try NTN import
NTN_MODELS_AVAILABLE = False
try:
    from sionna.phy.channel.tr38811 import DenseUrban, Antenna, AntennaArray
    from sionna.phy.channel.tr38811 import utils as tr811_utils
    NTN_MODELS_AVAILABLE = True
    print("âœ“ NTN (TR 38.811) models loaded")
except Exception:
    print("âš ï¸ NTN not available, using Rayleigh fallback")


class ISACSystem:
    """ISAC System with optimized topology caching for fast dataset generation."""
    
    def __init__(self):
        # RF/OFDM params from settings
        self.CARRIER_FREQUENCY = CARRIER_FREQUENCY
        self.SUBCARRIER_SPACING = SUBCARRIER_SPACING
        self.FFT_SIZE = FFT_SIZE
        self.NUM_OFDM_SYMBOLS = NUM_OFDM_SYMBOLS
        self.CYCLIC_PREFIX_LENGTH = CYCLIC_PREFIX_LENGTH
        
        # Antenna config
        self.SAT_ANTENNA = SAT_ANTENNA
        self.UT_ANTENNA = UT_ANTENNA
        
        self.NUM_SAT_BEAMS = 1
        self.NUM_UT = 1
        self.NUM_RX_ANT = 1
        self.NUM_TX_ANT = (
            self.SAT_ANTENNA["num_rows"] * 
            self.SAT_ANTENNA["num_cols"] * 2
        )
        
        # NTN geometry
        self.SCENARIO_TOPOLOGY = SCENARIO_TOPOLOGY
        self.SAT_HEIGHT = SAT_HEIGHT
        self.ELEVATION_ANGLE = ELEVATION_ANGLE
        
        # MCS/LDPC
        self.NUM_BITS_PER_SYMBOL = NUM_BITS_PER_SYMBOL
        self.CODERATE = CODERATE
        self.k = LDPC_K
        self.n = LDPC_N
        
        # Sionna components
        self.binary_source = BinarySource()
        self.mapper = Mapper("qam", self.NUM_BITS_PER_SYMBOL)
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
        
        self.modulator = OFDMModulator(self.rg.cyclic_prefix_length)
        l_min, l_max = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.demodulator = OFDMDemodulator(
            fft_size=self.rg.fft_size,
            l_min=l_min,
            l_max=l_max,
            cyclic_prefix_length=self.rg.cyclic_prefix_length
        )
        
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder)
        
        # Channel model
        self._init_channel()
        
        # Topology cache
        self.topology_cache = []
        
        # ✅ NEW: STNN models for fast localization
        self.stnn_estimator = None
        self._init_stnn_models()
        
        print("[ISAC] System initialized successfully")
    
    def _init_channel(self):
        """Initialize channel model (NTN or Rayleigh)."""
        if NTN_MODELS_AVAILABLE and USE_NTN_IF_AVAILABLE:
            print("[ISAC] Using TR 38.811 DenseUrban (NTN)")
            self.ut_array = Antenna(
                polarization=self.UT_ANTENNA["polarization"],
                polarization_type=self.UT_ANTENNA["polarization_type"],
                antenna_pattern="38.901",
                carrier_frequency=self.CARRIER_FREQUENCY
            )
            self.bs_array = AntennaArray(
                num_rows=self.SAT_ANTENNA["num_rows"],
                num_cols=self.SAT_ANTENNA["num_cols"],
                polarization=self.SAT_ANTENNA["polarization"],
                polarization_type=self.SAT_ANTENNA["polarization_type"],
                antenna_pattern="38.901",
                carrier_frequency=self.CARRIER_FREQUENCY
            )
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
                print(f"[WARN] NTN topology init failed: {e}")
                self._fallback_rayleigh()
        else:
            self._fallback_rayleigh()
    
    def _fallback_rayleigh(self):
        """Fallback to Rayleigh channel."""
        print("[ISAC] Using Rayleigh Block Fading")
        self.CHANNEL_MODEL = RayleighBlockFading(
            num_rx=self.NUM_UT,
            num_rx_ant=self.NUM_RX_ANT,
            num_tx=self.NUM_SAT_BEAMS,
            num_tx_ant=self.NUM_TX_ANT
        )
    
    def _init_stnn_models(self):
        """
        Initialize STNN models for fast TDOA/FDOA estimation.
        
        ✅ NEW: Loads trained STNN models from paper's method
        """
        from config.settings import (
            USE_STNN_LOCALIZATION,
            STNN_TDOA_MODEL_PATH,
            STNN_FDOA_MODEL_PATH,
            STNN_ERROR_STATS_PATH
        )
        
        if not USE_STNN_LOCALIZATION:
            print("[ISAC] STNN localization disabled")
            return
        
        try:
            import pickle
            from model.stnn_localization import STNNEstimator
            
            # Load STNN estimator
            self.stnn_estimator = STNNEstimator(
                tdoa_model_path=STNN_TDOA_MODEL_PATH,
                fdoa_model_path=STNN_FDOA_MODEL_PATH
            )
            
            # Load error statistics (from validation set)
            if os.path.exists(STNN_ERROR_STATS_PATH):
                with open(STNN_ERROR_STATS_PATH, 'rb') as f:
                    stats = pickle.load(f)
                
                self.stnn_estimator.update_error_statistics(
                    tdoa_std=stats['tdoa_std'],
                    fdoa_std=stats['fdoa_std']
                )
                
                print(f"[ISAC] ✓ STNN models loaded successfully")
            else:
                print(f"[ISAC] ⚠️  STNN error stats not found, using defaults")
        
        except Exception as e:
            print(f"[ISAC] ⚠️  STNN initialization failed: {e}")
            print("[ISAC] → Falling back to traditional GCC-PHAT")
            self.stnn_estimator = None
    
    def precompute_topologies(self, count=1000):
        """
        Pre-generate topology cache for fast reuse.
        
        âœ… OPTIMIZED: Generate 1000 topologies once (~2 min)
        Instead of generating per-sample (~0.5s each = 25 min total for 3000 samples)
        
        Args:
            count: Number of topologies to cache (default 1000)
        """
        if not NTN_MODELS_AVAILABLE or not USE_NTN_IF_AVAILABLE:
            print("[ISAC] Skipping topology cache (not using NTN)")
            return
        
        self.topology_cache.clear()
        print(f"[ISAC] Pre-generating {count} NTN topologies (this takes ~2 min)...")
        
        import time
        start = time.time()
        
        for i in range(count):
            # Vary altitude randomly for diversity (500-650 km)
            altitude = np.random.uniform(500e3, 650e3)
            
            try:
                topo = tr811_utils.gen_single_sector_topology(
                    batch_size=1,
                    num_ut=self.NUM_UT,
                    scenario=self.SCENARIO_TOPOLOGY,
                    elevation_angle=self.ELEVATION_ANGLE,
                    bs_height=float(altitude)
                )
                self.topology_cache.append(topo)
            except Exception as e:
                if i == 0:  # Only warn on first failure
                    print(f"[WARN] Topology generation failed: {e}")
                continue
            
            # Progress update every 200 topologies
            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (count - i - 1)
                print(f"  Progress: {i+1}/{count} topologies ({elapsed:.1f}s elapsed, ETA: {eta:.1f}s)")
        
        elapsed = time.time() - start
        print(f"[ISAC] âœ“ Cached {len(self.topology_cache)} topologies in {elapsed:.1f}s")
        print(f"[ISAC] â†’ Speedup: ~{count * 0.5 / 60:.1f} min saved during dataset generation!")
    
    def set_cached_topology(self, idx=None):
        """
        Set topology from cache (fast!).
        
        Args:
            idx: Index in cache (if None, random)
        """
        if not self.topology_cache:
            return  # No cache available
        
        if idx is None:
            idx = np.random.randint(0, len(self.topology_cache))
        else:
            idx = int(idx) % len(self.topology_cache)
        
        try:
            self.CHANNEL_MODEL.set_topology(*self.topology_cache[idx])
        except Exception:
            pass  # Silent fail (channel may not support topology)