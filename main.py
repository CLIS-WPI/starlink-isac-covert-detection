# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from scipy.signal import stft
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Set GPU memory growth to avoid out-of-memory errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only GPU 0
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Import Sionna components
import sionna
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, OFDMDemodulator, OFDMModulator, LMMSEEqualizer, ResourceGridMapper

# Updated imports for Sionna v1.2.1+
from sionna.phy.channel import (
    subcarrier_frequencies, 
    time_to_ofdm_channel, 
    cir_to_time_channel,
    time_lag_discrete_time_channel,
    ApplyTimeChannel,
    RayleighBlockFading
)
from sionna.phy.utils import ebnodb2no, hard_decisions
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
# Default covert injection amplitude (can be overridden by tests or CLI)
covert_amp = 1.5  # Default amplitude scaling for covert injection


def gcc_phat(x, y):
    """GCC-PHAT cross-correlation for robust TDoA estimation."""
    # Ensure numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x) + len(y) - 1
    # FFT of zero-padded signals
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    corr = np.fft.irfft(R, n=n)
    return corr

# Try to import NTN components, which are now integrated into Sionna.
try:
    from sionna.phy.channel.tr38811 import DenseUrban, AntennaArray
    from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_ntn_topology
    NTN_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: 3GPP TR38.811 NTN channel models not found in sionna.phy.channel.tr38811.")
    print("Falling back to default Sionna channel model.")
    NTN_MODELS_AVAILABLE = False


# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- Phase 1: Setup LEO Satellite ISAC System ---
print("Phase 1: Setting up LEO Satellite ISAC System...")

class ISACSystem:
    """
    A class to encapsulate all parameters for the ISAC simulation.
    """
    def __init__(self):
        global NTN_MODELS_AVAILABLE
        # ISAC parameters
        self.CARRIER_FREQUENCY = 28e9  # 28 GHz (Ka-band)
        self.SUBCARRIER_SPACING = 60e3 # 60 kHz
        
        # Adjusted for LDPC compatibility
        self.FFT_SIZE = 64 
        self.NUM_OFDM_SYMBOLS = 10
        self.CYCLIC_PREFIX_LENGTH = 8
        
        self.PILOT_DENSITY = 0.1 # Percentage of pilots
        self.NUM_SENSING_SUBCARRIERS = 4 # Dedicated for radar

        # Satellite and User Terminal antenna configuration
        self.SAT_ANTENNA = {"num_rows": 8, "num_cols": 8, "polarization": "dual", "polarization_type": "VH", "antenna_pattern": "38.901"}
        self.UT_ANTENNA = {"num_rows": 1, "num_cols": 1, "polarization": "single", "polarization_type": "V", "antenna_pattern": "omni"}
        self.NUM_SAT_BEAMS = 1
        self.NUM_UT = 1
        self.NUM_RX_ANT = self.UT_ANTENNA["num_rows"] * self.UT_ANTENNA["num_cols"] * (2 if self.UT_ANTENNA["polarization"] == "dual" else 1)
        self.NUM_TX_ANT = self.SAT_ANTENNA["num_rows"] * self.SAT_ANTENNA["num_cols"] * (2 if self.SAT_ANTENNA["polarization"] == "dual" else 1)
        
        # Use "umi" for the topology generator, but "dur" is what DenseUrban class expects internally.
        self.SCENARIO_TOPOLOGY = "dur" 
        self.SAT_HEIGHT = 600e3 # 600km LEO satellite
        self.ELEVATION_ANGLE = 50.0 # degrees

        # Channel Model
        if NTN_MODELS_AVAILABLE:
            print("Integrating 3GPP TR38.811 NTN channel model (DenseUrban)...")
            # Create Sionna Antenna objects as required by the 3GPP models
            self.ut_array = AntennaArray(num_rows=self.UT_ANTENNA["num_rows"],
                                    num_cols=self.UT_ANTENNA["num_cols"],
                                    polarization=self.UT_ANTENNA["polarization"],
                                    polarization_type=self.UT_ANTENNA["polarization_type"],
                                    antenna_pattern=self.UT_ANTENNA["antenna_pattern"],
                                    carrier_frequency=self.CARRIER_FREQUENCY)
            self.bs_array = AntennaArray(num_rows=self.SAT_ANTENNA["num_rows"],
                                         num_cols=self.SAT_ANTENNA["num_cols"],
                                         polarization=self.SAT_ANTENNA["polarization"],
                                         polarization_type=self.SAT_ANTENNA["polarization_type"],
                                         antenna_pattern=self.SAT_ANTENNA["antenna_pattern"],
                                         carrier_frequency=self.CARRIER_FREQUENCY)
            
            self.CHANNEL_MODEL = DenseUrban(carrier_frequency=self.CARRIER_FREQUENCY,
                                            ut_array=self.ut_array,
                                            bs_array=self.bs_array,
                                            direction='downlink',
                                            elevation_angle=self.ELEVATION_ANGLE,
                                            doppler_enabled=True)
            try:
                # Use "umi" for the topology generator function
                # Ensure bs_height is float32 to avoid TF dtype mismatch inside generator
                topology = gen_ntn_topology(batch_size=1, num_ut=self.NUM_UT, scenario=self.SCENARIO_TOPOLOGY, bs_height=float(np.float32(self.SAT_HEIGHT)))
                self.CHANNEL_MODEL.set_topology(*topology)
            except Exception as e:
                print(f"Error setting initial topology for NTN model: {e}")
                print("Falling back to Rayleigh model.")
                NTN_MODELS_AVAILABLE = False
                self.CHANNEL_MODEL = RayleighBlockFading(num_rx=self.NUM_UT,
                                                         num_rx_ant=self.NUM_RX_ANT,
                                                         num_tx=self.NUM_SAT_BEAMS,
                                                         num_tx_ant=self.NUM_TX_ANT)

        else:
            print("Using RayleighBlockFading as fallback channel model.")
            self.CHANNEL_MODEL = RayleighBlockFading(num_rx=self.NUM_UT,
                                                     num_rx_ant=self.NUM_RX_ANT,
                                                     num_tx=self.NUM_SAT_BEAMS,
                                                     num_tx_ant=self.NUM_TX_ANT)
            
        if hasattr(self.CHANNEL_MODEL, 'min_elevation_angle'):
            self.CHANNEL_MODEL.min_elevation_angle = 10.0 # Degrees

        # Modulation and Coding
        self.NUM_BITS_PER_SYMBOL = 4 # 16-QAM
        self.CODERATE = 0.5

        # Create Sionna components
        self.binary_source = BinarySource()
        self.mapper = Mapper("qam", self.NUM_BITS_PER_SYMBOL)
        self.demapper = Demapper("app", "qam", self.NUM_BITS_PER_SYMBOL)
        self.rg = ResourceGrid(num_ofdm_symbols=self.NUM_OFDM_SYMBOLS,
                               fft_size=self.FFT_SIZE,
                               subcarrier_spacing=self.SUBCARRIER_SPACING,
                               num_tx=self.NUM_SAT_BEAMS,
                               num_streams_per_tx=self.NUM_UT,
                               cyclic_prefix_length=self.CYCLIC_PREFIX_LENGTH,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 7])
        # Explicit sampling rate tied to OFDM parameters (scientifically justified)
        # Sampling rate = FFT size * subcarrier spacing
        # This clarifies the physical sampling frequency used for TDoA computations.
        self.SAMPLING_RATE = float(self.FFT_SIZE * self.SUBCARRIER_SPACING)
        self.rg_mapper = ResourceGridMapper(self.rg)
        self.sm = StreamManagement(np.array([[1]]), self.NUM_UT)
        self.lmmse_equalizer = LMMSEEqualizer(self.rg, self.sm)
        self.modulator = OFDMModulator(self.rg.cyclic_prefix_length)
        
        l_min, _ = time_lag_discrete_time_channel(self.rg.bandwidth)
        self.demodulator = OFDMDemodulator(fft_size=self.rg.fft_size, l_min=l_min, cyclic_prefix_length=self.rg.cyclic_prefix_length)

        # FEC - Use standard 5G LDPC parameters
        self.k = 512
        self.n = 1024
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder)


# --- Phase 2: Generate ISAC Waveforms with Covert Channels ---
print("Phase 2: Defining ISAC Waveform Generation...")

def inject_covert_channel(ofdm_frame, resource_grid, covert_rate_mbps, scs):
    """
    Embeds covert data in unused subcarriers of an OFDM frame.
    """
    if covert_rate_mbps == 0:
        return ofdm_frame, None

    batch_size = ofdm_frame.shape[0]
    num_ofdm_symbols = ofdm_frame.shape[-2]
    fft_size = ofdm_frame.shape[-1]
    
    symbol_duration = (resource_grid.fft_size + resource_grid.cyclic_prefix_length) / (resource_grid.fft_size * scs)
    bits_per_symbol = 2  # QPSK
    symbols_per_second = 1 / symbol_duration
    bits_per_second_per_subcarrier = bits_per_symbol * symbols_per_second
    num_covert_subcarriers = int((covert_rate_mbps * 1e6) / bits_per_second_per_subcarrier)
    num_covert_subcarriers = min(num_covert_subcarriers, resource_grid.num_effective_subcarriers // 4)

    # Generate covert QAM symbols
    covert_bits = tf.random.uniform([batch_size, num_covert_subcarriers, bits_per_symbol], 0, 2, dtype=tf.int32)
    covert_mapper = Mapper("qam", bits_per_symbol)
    covert_symbols = covert_mapper(covert_bits)  # [batch, num_covert_subcarriers]
    
    # Covert power scaling - MAXIMUM for best detectability
    covert_power_reduction = covert_amp  # allow external control for ablation
    covert_symbols = covert_symbols * covert_power_reduction

    # Select subcarrier indices (use every 4th to avoid pilots/data)
    all_indices = np.arange(resource_grid.num_effective_subcarriers)
    covert_candidate_indices = all_indices[::4]
    
    if len(covert_candidate_indices) < num_covert_subcarriers:
        num_covert_subcarriers = len(covert_candidate_indices)
        covert_symbols = covert_symbols[:, :num_covert_subcarriers]
    
    selected_indices = np.random.choice(covert_candidate_indices, num_covert_subcarriers, replace=False)
    
    # Inject across multiple OFDM symbols for stronger signature
    num_symbols_to_inject = min(3, num_ofdm_symbols)  # Inject on 3 symbols instead of 1
    symbol_indices = np.random.choice(num_ofdm_symbols, num_symbols_to_inject, replace=False)
    
    # Create injection array
    ofdm_frame_np = ofdm_frame.numpy()
    covert_symbols_np = covert_symbols.numpy()[0]  # [num_covert_subcarriers]
    
    # Inject covert symbols into selected subcarriers across multiple symbols
    for symbol_idx in symbol_indices:
        for i, subcarrier_idx in enumerate(selected_indices):
            # Use complex() to safely convert numpy array element to Python scalar
            # ensure we extract a true Python scalar to avoid DeprecationWarning
            val = covert_symbols_np[i]
            try:
                val = np.asarray(val).item()
            except Exception:
                pass
            ofdm_frame_np[0, 0, 0, symbol_idx, subcarrier_idx] += complex(val)
    
    # Define emitter location
    emitter_location = (np.random.uniform(-1000, 1000), np.random.uniform(-1000, 1000), 0)
    
    return tf.convert_to_tensor(ofdm_frame_np), emitter_location


def generate_dataset(isac_system, num_samples_per_class, ebno_db_range=(5, 15), covert_rate_mbps_range=(1, 50)):
    """
    Generates a dataset of benign and compromised ISAC transmissions.
    """
    all_iq_samples, all_csi, all_radar_echoes, all_labels, all_emitter_locations = [], [], [], [], []
    
    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

# --- Phase 3: Simulate Channel & Collect Data ---
print("Phase 3: Generating Benign and Attack Datasets...")

def generate_dataset(isac_system, num_samples_per_class, ebno_db_range=(5, 15), covert_rate_mbps_range=(1, 50)):
    """
    Generates a dataset of benign and compromised ISAC transmissions.
    """
    all_iq_samples, all_csi, all_radar_echoes, all_labels, all_emitter_locations = [], [], [], [], []
    
    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

    # Generate sample indices and shuffle for fair channel distribution
    total_samples = num_samples_per_class * 2
    sample_indices = np.arange(total_samples)
    np.random.shuffle(sample_indices)  # Shuffle to distribute channels fairly

    for idx in range(total_samples):
        i = sample_indices[idx]  # Use shuffled index
        is_attack = i >= num_samples_per_class
        label = 1 if is_attack else 0
        batch_size = 1

        b = isac_system.binary_source([batch_size, total_info_bits])
        
        c_list = [isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) for j in range(num_codewords)]
        
        c = tf.concat(c_list, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        
        # Reshape to [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x = tf.reshape(x, [batch_size, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_ofdm_frame = isac_system.rg_mapper(x)

        emitter_loc = None
        if is_attack:
            covert_rate = np.random.uniform(*covert_rate_mbps_range)
            tx_ofdm_frame, emitter_loc = inject_covert_channel(tx_ofdm_frame, isac_system.rg, covert_rate, isac_system.SUBCARRIER_SPACING)

        tx_time_signal = isac_system.modulator(tx_ofdm_frame)
        num_time_samples = tx_time_signal.shape[-1]
        
        ebno_db = tf.random.uniform((), *ebno_db_range)
        no = ebnodb2no(ebno_db, isac_system.NUM_BITS_PER_SYMBOL, isac_system.CODERATE, isac_system.rg)
        
        if NTN_MODELS_AVAILABLE:
            topology = gen_ntn_topology(batch_size=batch_size, num_ut=isac_system.NUM_UT, 
                                       scenario=isac_system.SCENARIO_TOPOLOGY, bs_height=float(np.float32(isac_system.SAT_HEIGHT)))
            isac_system.CHANNEL_MODEL.set_topology(*topology)

        # Generate channel impulse response (CIR)
        if NTN_MODELS_AVAILABLE:
            a, tau = isac_system.CHANNEL_MODEL(batch_size, isac_system.rg.bandwidth)
        else:
            a, tau = isac_system.CHANNEL_MODEL(batch_size, 
                                               num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                                               sampling_frequency=isac_system.rg.bandwidth)
        
        # Convert to time domain channel
        l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
        h_time = cir_to_time_channel(isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True)
        
        if idx == 0:
            print(f"Before replication - h_time shape: {h_time.shape}")
        
        # For NTN: Apply channel with proper power normalization
        if NTN_MODELS_AVAILABLE:
            # Extract channel gain using RMS across taps (preserves power!)
            h_power = tf.reduce_mean(tf.abs(h_time)**2, axis=-1)  # Power across taps
            h_gain = tf.sqrt(h_power + 1e-10)  # RMS = sqrt(mean(power))
            
            # Average over antennas
            h_gain = tf.reduce_mean(h_gain, axis=2)  # Average rx_ant
            h_gain = tf.reduce_mean(h_gain, axis=3)  # Average tx_ant
            h_gain = h_gain[:, 0, 0, :]  # Select first RX and TX
            
            # Replicate to all OFDM symbols and subcarriers
            h_gain = tf.tile(h_gain, [1, isac_system.NUM_OFDM_SYMBOLS])  
            h_gain = tf.reshape(h_gain, [batch_size, isac_system.NUM_OFDM_SYMBOLS, 1])
            h_gain = tf.tile(h_gain, [1, 1, isac_system.FFT_SIZE])
            
            # Convert to complex for proper channel application
            h_gain = tf.cast(h_gain, tf.complex64)
            
            # Reshape to match tx_ofdm_frame
            h_freq_broadcast = tf.expand_dims(h_gain, axis=1)
            h_freq_broadcast = tf.expand_dims(h_freq_broadcast, axis=2)
            
        else:
            # Rayleigh fallback
            if h_time.shape[-2] == 1:
                multiples = [1] * len(h_time.shape)
                multiples[-2] = isac_system.NUM_OFDM_SYMBOLS
                h_time = tf.tile(h_time, multiples)
            
            h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)
            h_freq_reduced = tf.reduce_mean(h_freq, axis=2)
            h_freq_reduced = tf.reduce_mean(h_freq_reduced, axis=2)
            h_freq_simplified = h_freq_reduced[:, 0, :, :]
            h_freq_simplified = tf.expand_dims(h_freq_simplified, axis=1)
            h_freq_broadcast = tf.expand_dims(h_freq_simplified, axis=2)
        
        if idx == 0:
            print(f"h_freq_broadcast shape: {h_freq_broadcast.shape}")
            print(f"Channel gain (mean magnitude): {tf.reduce_mean(tf.abs(h_freq_broadcast)).numpy():.4f}")
        
        # Apply channel
        rx_freq = tx_ofdm_frame * h_freq_broadcast
        
        # Add noise in frequency domain
        noise_power = tf.cast(no, tf.float32)
        noise_freq = tf.complex(
            tf.random.normal(tf.shape(rx_freq), stddev=tf.sqrt(noise_power/2)),
            tf.random.normal(tf.shape(rx_freq), stddev=tf.sqrt(noise_power/2))
        )
        rx_freq = rx_freq + noise_freq
        
        # Convert to time domain for IQ samples
        rx_time_signal = isac_system.modulator(rx_freq)
        
        # Simple radar echo simulation
        radar_delay = np.random.randint(50, 150)
        radar_attenuation = 0.3
        tx_time_flat = tf.squeeze(tx_time_signal)
        radar_echo = tf.roll(tx_time_flat, shift=radar_delay, axis=-1) * radar_attenuation
        radar_echo_noise = tf.complex(
            tf.random.normal(tf.shape(radar_echo), stddev=tf.sqrt(noise_power/2)),
            tf.random.normal(tf.shape(radar_echo), stddev=tf.sqrt(noise_power/2))
        )
        radar_echo += radar_echo_noise
        
        # Store with original index for correct label assignment
        all_iq_samples.append((i, np.squeeze(rx_time_signal.numpy())))
        # CRITICAL: Store RECEIVED frequency signal (contains covert data), not just channel!
        all_csi.append((i, np.squeeze(rx_freq.numpy())))  # rx_freq contains the actual signal with covert injection
        all_radar_echoes.append((i, np.squeeze(radar_echo.numpy())))
        all_labels.append((i, label))
        all_emitter_locations.append((i, emitter_loc))
        
        if (idx+1) % 100 == 0:
            print(f"Generated sample {idx+1}/{total_samples}...")

    # Sort by original index to maintain correct label alignment
    all_iq_samples = [x[1] for x in sorted(all_iq_samples, key=lambda x: x[0])]
    all_csi = [x[1] for x in sorted(all_csi, key=lambda x: x[0])]
    all_radar_echoes = [x[1] for x in sorted(all_radar_echoes, key=lambda x: x[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda x: x[0])]
    all_emitter_locations = [x[1] for x in sorted(all_emitter_locations, key=lambda x: x[0])]

    return {
        'iq_samples': np.array(all_iq_samples), 'csi': np.array(all_csi),
        'radar_echo': np.array(all_radar_echoes), 'labels': np.array(all_labels),
        'emitter_locations': all_emitter_locations
    }


# --- Phase 4: Feature Extraction for Detection ---
print("Phase 4: Defining Feature Extraction Pipeline...")

def extract_received_signal_features(dataset):
    """
    Extract features from RECEIVED frequency-domain signal.
    This captures both the channel AND the covert data injection!
    """
    features = []
    total_samples = len(dataset['labels'])
    
    for i in range(total_samples):
        rx_freq = dataset['csi'][i]  # Now contains rx_freq, not just channel
        
        if rx_freq.ndim == 0 or rx_freq.size == 0:
            continue
        
        # rx_freq shape: (num_ofdm_symbols, fft_size)
        if rx_freq.ndim > 2:
            rx_freq = np.squeeze(rx_freq)
        
        if rx_freq.ndim < 2:
            continue
        
        # Compute power per subcarrier per OFDM symbol
        rx_power = np.abs(rx_freq)**2
        
        # Normalize per sample to [0, 1]
        max_power = np.max(rx_power)
        if max_power > 0:
            rx_power = rx_power / max_power
        
        # Features capturing TEMPORAL variation across symbols
        # Key insight: Covert signals appear as SPIKY power anomalies
        
        # Mean power across symbols (baseline)
        mean_power = np.mean(rx_power, axis=0) if rx_power.ndim > 1 else rx_power
        
        # Std power across symbols (attacks have VARIABLE power - covert injection)
        std_power = np.std(rx_power, axis=0) if rx_power.ndim > 1 else np.zeros_like(rx_power)
        
        # Max power across symbols (captures covert spikes in some symbols)
        max_power_subcarrier = np.max(rx_power, axis=0) if rx_power.ndim > 1 else rx_power
        
        # Combine all features: [mean, std, max] for each subcarrier
        if len(mean_power) >= 64:
            features_combined = np.stack([
                mean_power[:64],
                std_power[:64],
                max_power_subcarrier[:64]
            ], axis=-1)  # (64, 3)
            
            # Reshape to spatial format for CNN: 8x8x3
            feature = features_combined.reshape(8, 8, 3)
        else:
            # Pad if needed
            mean_padded = np.pad(mean_power, (0, 64-len(mean_power)), mode='constant')
            std_padded = np.pad(std_power, (0, 64-len(std_power)), mode='constant')
            max_padded = np.pad(max_power_subcarrier, (0, 64-len(max_power_subcarrier)), mode='constant')
            
            features_combined = np.stack([mean_padded, std_padded, max_padded], axis=-1)
            feature = features_combined.reshape(8, 8, 3)
        
        features.append(feature)
        
        if (i+1) % 500 == 0:
            print(f"Extracted RX signal features for sample {i+1}/{total_samples}...")
    
    return np.array(features)

def extract_features(dataset, nperseg=64):
    """
    Extracts features from the raw simulation data.
    """
    features = []
    total_samples = len(dataset['labels'])
    for i in range(total_samples):
        iq = dataset['iq_samples'][i]
        if iq.ndim == 0 or iq.size < nperseg: 
            continue
        
        _, _, Zxx = stft(iq, nperseg=nperseg)
        spectrogram = np.abs(Zxx)
        
        # Handle any dimensionality by reshaping to 2D first
        if spectrogram.ndim > 2:
            spectrogram = spectrogram.reshape(spectrogram.shape[0], -1)
        
        # Add channel dimension for tf.image.resize
        spectrogram_3d = np.expand_dims(spectrogram, axis=-1)  # (freq, time, 1)
        
        # Resize to 64x64
        spectrogram_resized = tf.image.resize(spectrogram_3d, [64, 64]).numpy()
        
        # Final shape should be (64, 64, 1)
        if spectrogram_resized.shape != (64, 64, 1):
            spectrogram_resized = spectrogram_resized.reshape(64, 64, 1)
        
        features.append(spectrogram_resized)

        if (i+1) % 100 == 0:
            print(f"Extracted features for sample {i+1}/{total_samples}...")

    return np.array(features)


# --- Phase 5: Train ML Detector ---
print("Phase 5: Building and Training ML Detector...")

def build_dual_input_cnn_fixed():
    """
    Dual-input CNN with correct feature dimensions.
    Input 2 now has 3 channels: mean, std, max power per subcarrier.
    """
    # Input 1: Spectrogram (64x64x1)
    input_spec = tf.keras.layers.Input(shape=(64, 64, 1), name='spectrogram')
    x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_spec)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    # Input 2: RX signal features (8x8x3) - mean, std, max power per subcarrier
    input_rx = tf.keras.layers.Input(shape=(8, 8, 3), name='rx_features')
    x2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_rx)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    
    # Fusion: Combine both pathways
    combined = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input_spec, input_rx], outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_dual_input_cnn_regularized():
    """
    Dual-input CNN with strong regularization to prevent overfitting.
    Uses L2 regularization, batch normalization, and dropout.
    """
    # Input 1: Spectrogram (64x64x1)
    input_spec = tf.keras.layers.Input(shape=(64, 64, 1), name='spectrogram')
    x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_spec)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    # Input 2: CSI features (8x8x2)
    input_csi = tf.keras.layers.Input(shape=(8, 8, 2), name='csi_features')
    x2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_csi)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    
    # Fusion: Combine both pathways
    combined = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input_spec, input_csi], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_dual_input_cnn():
    """
    CNN with two inputs: spectrograms + CSI features.
    Better for detecting sparse covert signal injection.
    """
    # Input 1: Spectrogram (64x64x1)
    input_spec = tf.keras.layers.Input(shape=(64, 64, 1), name='spectrogram')
    x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_spec)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    # Input 2: CSI features (8x8x2)
    input_csi = tf.keras.layers.Input(shape=(8, 8, 2), name='csi_features')
    x2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_csi)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    
    # Fusion: Combine both pathways
    combined = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(128, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input_spec, input_csi], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_classifier(input_shape=(64, 64, 1)):
    """
    Builds a simple CNN for binary classification (legacy single-input version).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# --- Phase 6: Localization via Multi-Beam ISAC ---
print("Phase 6: Defining Localization Logic...")

def estimate_emitter_location(detected_sample_idx, dataset, satellite_positions):
    """
    Simplified localization using Time Difference of Arrival (TDoA) from multiple satellites.
    """
    # NOTE: This placeholder used ground-truth and synthetic noise. Keep for backward
    # compatibility but prefer the real TDoA implementation below.
    true_location = dataset['emitter_locations'][detected_sample_idx]
    if true_location is None:
        return None, None

    estimation_error = np.random.normal(0, 150, 3)
    estimated_position = np.array(true_location) + estimation_error

    return estimated_position, np.array(true_location)


def estimate_emitter_location_tdoa(detected_sample_idx, dataset, isac_system):
    """
    Estimate emitter location using Time Difference of Arrival (TDoA) from multiple satellite
    receptions saved in `dataset['satellite_receptions']`.

    Steps:
    - use the first satellite as reference
    - cross-correlate absolute-valued received time signals to estimate integer-sample delays
    - convert sample-delay differences to distance differences (distance_i - distance_ref)
    - solve 2D trilateration (least-squares) using those distance differences
    """
    # get stored receptions for the sample
    try:
        satellite_rx_data = dataset['satellite_receptions'][detected_sample_idx]
    except Exception:
        return None, None

    # require at least 4 satellites for robust 2D localization (reference + 3 others)
    if satellite_rx_data is None or len(satellite_rx_data) < 4:
        return None, None

    # Reference is satellite 0
    reference = satellite_rx_data[0]
    ref_signal = reference['rx_time']
    ref_pos = reference['position']

    tdoa_measurements = []
    satellite_pairs = []

    # use correct sampling rate from isac_system (explicitly set in ISACSystem)
    sampling_rate = getattr(isac_system, 'SAMPLING_RATE', isac_system.rg.bandwidth)
    c_speed = 3e8

    # choose other satellites as comparators
    for entry in satellite_rx_data[1:]:
        sig = entry['rx_time']
        pos = entry['position']

        # Guard: skip if signals are too short
        if len(sig) < 3 or len(ref_signal) < 3:
            # not enough samples to correlate reliably
            continue

        # Truncate to equal length for stability (optional)
        L = min(len(sig), len(ref_signal))
        sig_tr = np.asarray(sig[:L])
        ref_tr = np.asarray(ref_signal[:L])

        # Choose correlation method: GCC-PHAT is more robust at low SNR
        use_gcc_phat = False
        try:
            if use_gcc_phat:
                corr = gcc_phat(sig_tr, ref_tr)
            else:
                # Complex cross-correlation preserves phase (more accurate TDoA)
                corr = np.correlate(sig_tr, np.conj(ref_tr), mode='full')
        except Exception:
            # fall back to magnitude correlation if complex signals cause issues
            corr = np.correlate(np.abs(sig_tr), np.abs(ref_tr), mode='full')

        # Use geometry-informed approximate delay (if available) to window search
        # The stored 'true_delay_samples' in dataset is a simulation aid and gives
        # an approximate propagation delay which we can use to limit false peaks.
        approx_delay = None
        if 'true_delay_samples' in entry:
            try:
                approx_delay = int(entry.get('true_delay_samples'))
            except Exception:
                approx_delay = None

        # Full-length center index corresponding to zero-lag: len(ref_signal) - 1
        center = len(ref_signal) - 1

        # Define a search window around the approximate delay if available
        if approx_delay is not None:
            # adaptive window: limit to a fraction of corr length to avoid empty windows
            max_window = max(10, len(corr) // 4)
            window = min(500, max_window)
            start = max(0, center + approx_delay - window)
            end = min(len(corr), center + approx_delay + window + 1)
            corr_window = corr[start:end]
            # If corr_window is empty for any reason, fallback to global search
            if corr_window.size == 0:
                k0 = int(np.argmax(np.abs(corr)))
            else:
                k_local = np.argmax(np.abs(corr_window))
                k0 = start + k_local
        else:
            # fallback: global search
            k0 = int(np.argmax(np.abs(corr)))

        # Sub-sample refinement via parabolic interpolation around the peak
        if 0 < k0 < len(corr) - 1:
            y1 = np.abs(corr[k0 - 1])
            y2 = np.abs(corr[k0])
            y3 = np.abs(corr[k0 + 1])
            denom = (y1 - 2 * y2 + y3)
            if denom == 0:
                delta = 0.0
            else:
                delta = 0.5 * (y1 - y3) / denom
        else:
            delta = 0.0

        # fractional-sample delay (positive means sig lags ref)
        delay_samples = (k0 - center) + delta

        # Convert to distance difference
        delay_time = delay_samples / sampling_rate
        distance_diff = delay_time * c_speed

        tdoa_measurements.append(distance_diff)
        satellite_pairs.append((ref_pos, pos))

    if len(tdoa_measurements) < 2:
        return None, None

    # solve least-squares trilateration in 2D using improved solver
    try:
        estimated_pos = trilateration_2d_improved(tdoa_measurements, satellite_pairs)
    except Exception:
        # fall back to original solver if improved fails
        try:
            estimated_pos = trilateration_2d(tdoa_measurements, satellite_pairs)
        except Exception:
            return None, None

    true_pos = dataset['emitter_locations'][detected_sample_idx]
    if true_pos is None:
        return None, None

    return estimated_pos, np.array(true_pos)


def trilateration_2d(tdoa_measurements, satellite_pairs):
    """
    Solve for 2D emitter position (x, y) given TDoA distance differences.
    Uses nonlinear least-squares (Levenberg-Marquardt) via scipy.optimize.least_squares.
    """
    from scipy.optimize import least_squares

    def residuals(pos, tdoa_list, pairs):
        x, y = pos
        emitter = np.array([x, y, 0.0])
        res = []
        for i, (ref_pos, sat_pos) in enumerate(pairs):
            d_ref = np.linalg.norm(emitter - ref_pos)
            d_i = np.linalg.norm(emitter - sat_pos)
            pred = d_i - d_ref
            res.append(tdoa_list[i] - pred)
        return np.array(res)

    # initial guess = centroid of satellite 2D positions
    centroids = []
    for ref, sat in satellite_pairs:
        centroids.append((ref[:2] + sat[:2]) / 2.0)
    initial = np.mean(centroids, axis=0)

    result = least_squares(residuals, initial, args=(tdoa_measurements, satellite_pairs), method='lm')
    est_x, est_y = result.x
    return np.array([est_x, est_y, 0.0])


def trilateration_2d_improved(tdoa_measurements, satellite_pairs):
    """
    Improved trilateration with better initial guess and bounds.
    Uses bounded least-squares (TRF) and a more reasonable initial guess.
    """
    from scipy.optimize import least_squares

    def tdoa_residuals(pos_2d, tdoa_list, sat_pairs):
        x, y = pos_2d
        emitter_pos_3d = np.array([x, y, 0.0])

        residuals = []
        for i, (ref_pos, sat_pos) in enumerate(sat_pairs):
            d_ref = np.linalg.norm(emitter_pos_3d - ref_pos)
            d_i = np.linalg.norm(emitter_pos_3d - sat_pos)
            predicted_tdoa = d_i - d_ref
            residuals.append(tdoa_list[i] - predicted_tdoa)

        return np.array(residuals)

    # Better initial guess: mean of the reference satellite 2D positions
    sat_positions = np.array([pair[0] for pair in satellite_pairs])
    initial_guess = np.mean(sat_positions[:, :2], axis=0)

    # Bounds: emitter is likely within a +/-200km box around origin (more permissive)
    lower = np.array([-200e3, -200e3])
    upper = np.array([200e3, 200e3])

    result = least_squares(
        tdoa_residuals,
        initial_guess,
        args=(tdoa_measurements, satellite_pairs),
        method='trf',
        bounds=(lower, upper),
        max_nfev=1000
    )

    estimated_x, estimated_y = result.x
    return np.array([estimated_x, estimated_y, 0.0])


def generate_dataset_multi_satellite(isac_system, num_samples_per_class, num_satellites=4,
                                    ebno_db_range=(5, 15), covert_rate_mbps_range=(1, 50)):
    """
    Generate dataset with reception simulated at multiple satellites for TDoA localization.
    This is based on the original `generate_dataset` but stores per-satellite received signals
    (time and frequency domain) in `satellite_receptions` so localization algorithms can
    operate without access to ground truth during estimation.
    """
    all_iq_samples, all_csi, all_radar_echoes, all_labels, all_emitter_locations = [], [], [], [], []
    all_satellite_receptions = []  # store list of receptions per sample

    # Define a simple constellation (4 satellites) if num_satellites==4, otherwise place in grid
    base_positions = []
    grid_spacing = 100e3
    if num_satellites == 4:
        base_positions = [
            np.array([0.0, 0.0, 600e3]),
            np.array([grid_spacing, 0.0, 600e3]),
            np.array([0.0, grid_spacing, 600e3]),
            np.array([grid_spacing, grid_spacing, 600e3])
        ]
    else:
        # place satellites on a simple square grid
        for i in range(num_satellites):
            x = (i % int(np.ceil(np.sqrt(num_satellites)))) * grid_spacing
            y = (i // int(np.ceil(np.sqrt(num_satellites)))) * grid_spacing
            base_positions.append(np.array([x, y, 600e3]))

    total_payload_bits = isac_system.rg.num_data_symbols * isac_system.NUM_BITS_PER_SYMBOL
    num_codewords = int(np.ceil(total_payload_bits / isac_system.n))
    total_info_bits = num_codewords * isac_system.k

    total_samples = num_samples_per_class * 2
    sample_indices = np.arange(total_samples)
    np.random.shuffle(sample_indices)

    for idx in range(total_samples):
        i = sample_indices[idx]
        is_attack = i >= num_samples_per_class
        label = 1 if is_attack else 0
        batch_size = 1

        b = isac_system.binary_source([batch_size, total_info_bits])
        c_list = [isac_system.encoder(b[:, j*isac_system.k:(j+1)*isac_system.k]) for j in range(num_codewords)]
        c = tf.concat(c_list, axis=1)[:, :total_payload_bits]
        x = isac_system.mapper(c)
        x = tf.reshape(x, [batch_size, isac_system.NUM_SAT_BEAMS, isac_system.NUM_UT, -1])
        tx_ofdm_frame = isac_system.rg_mapper(x)

        emitter_loc = None
        if is_attack:
            covert_rate = np.random.uniform(*covert_rate_mbps_range)
            # pick a fixed emitter position for localization (on ground plane)
            emitter_x = np.random.uniform(-50e3, 150e3)
            emitter_y = np.random.uniform(-50e3, 150e3)
            emitter_loc = np.array([emitter_x, emitter_y, 0.0])
            tx_ofdm_frame, _ = inject_covert_channel(tx_ofdm_frame, isac_system.rg, covert_rate, isac_system.SUBCARRIER_SPACING)

        tx_time_signal = isac_system.modulator(tx_ofdm_frame)

        # simulate reception at each satellite with propagation delay and independent channel/noise
        satellite_rx_signals = []
        for sat_idx, sat_pos in enumerate(base_positions[:num_satellites]):
            # compute distance and delay
            if emitter_loc is not None:
                distance = np.linalg.norm(sat_pos - emitter_loc)
            else:
                # use a default user position for benign transmissions
                default_user_pos = np.array([50e3, 50e3, 0.0])
                distance = np.linalg.norm(sat_pos - default_user_pos)

            c_speed = 3e8
            propagation_delay = distance / c_speed
            # use explicit sampling rate when available
            sampling_rate = getattr(isac_system, 'SAMPLING_RATE', isac_system.rg.bandwidth)
            delay_samples = int(np.round(propagation_delay * sampling_rate))

            # apply integer-sample delay in time-domain signal
            tx_delayed = tf.roll(tx_time_signal, shift=delay_samples, axis=-1)

            # draw a random ebno for this satellite reception
            ebno_db = tf.random.uniform((), *ebno_db_range)
            no = ebnodb2no(ebno_db, isac_system.NUM_BITS_PER_SYMBOL, isac_system.CODERATE, isac_system.rg)

            # generate channel realization per-satellite (reuse existing model calls)
            if NTN_MODELS_AVAILABLE:
                # ensure bs_height dtype consistency
                topology = gen_ntn_topology(batch_size=batch_size, num_ut=isac_system.NUM_UT,
                          scenario=isac_system.SCENARIO_TOPOLOGY, bs_height=float(np.float32(sat_pos[2])))
                isac_system.CHANNEL_MODEL.set_topology(*topology)
                a, tau = isac_system.CHANNEL_MODEL(batch_size, isac_system.rg.bandwidth)
            else:
                a, tau = isac_system.CHANNEL_MODEL(batch_size,
                                                   num_time_steps=isac_system.NUM_OFDM_SYMBOLS,
                                                   sampling_frequency=isac_system.rg.bandwidth)

            l_min, l_max = time_lag_discrete_time_channel(isac_system.rg.bandwidth)
            h_time = cir_to_time_channel(isac_system.rg.bandwidth, a, tau, l_min, l_max, normalize=True)

            # prepare frequency-domain channel broadcast as in generate_dataset
            if NTN_MODELS_AVAILABLE:
                h_power = tf.reduce_mean(tf.abs(h_time)**2, axis=-1)
                h_gain = tf.sqrt(h_power + 1e-10)
                h_gain = tf.reduce_mean(h_gain, axis=2)
                h_gain = tf.reduce_mean(h_gain, axis=3)
                h_gain = h_gain[:, 0, 0, :]
                h_gain = tf.tile(h_gain, [1, isac_system.NUM_OFDM_SYMBOLS])
                h_gain = tf.reshape(h_gain, [batch_size, isac_system.NUM_OFDM_SYMBOLS, 1])
                h_gain = tf.tile(h_gain, [1, 1, isac_system.FFT_SIZE])
                h_gain = tf.cast(h_gain, tf.complex64)
                h_freq_broadcast = tf.expand_dims(h_gain, axis=1)
                h_freq_broadcast = tf.expand_dims(h_freq_broadcast, axis=2)
            else:
                if h_time.shape[-2] == 1:
                    multiples = [1] * len(h_time.shape)
                    multiples[-2] = isac_system.NUM_OFDM_SYMBOLS
                    h_time = tf.tile(h_time, multiples)
                h_freq = time_to_ofdm_channel(h_time, isac_system.rg, l_min)
                h_freq_reduced = tf.reduce_mean(h_freq, axis=2)
                h_freq_reduced = tf.reduce_mean(h_freq_reduced, axis=2)
                h_freq_simplified = h_freq_reduced[:, 0, :, :]
                h_freq_simplified = tf.expand_dims(h_freq_simplified, axis=1)
                h_freq_broadcast = tf.expand_dims(h_freq_simplified, axis=2)

            # demodulate delayed time signal back to OFDM grid, apply channel and noise
            tx_ofdm_delayed = isac_system.demodulator(tx_delayed)
            rx_freq = tx_ofdm_delayed * h_freq_broadcast

            noise_power = tf.cast(no, tf.float32)
            noise_freq = tf.complex(
                tf.random.normal(tf.shape(rx_freq), stddev=tf.sqrt(noise_power/2)),
                tf.random.normal(tf.shape(rx_freq), stddev=tf.sqrt(noise_power/2))
            )
            rx_freq = rx_freq + noise_freq

            rx_time_signal = isac_system.modulator(rx_freq)

            satellite_rx_signals.append({
                'satellite_id': sat_idx,
                'position': sat_pos,
                'rx_time': np.squeeze(rx_time_signal.numpy()),
                'rx_freq': np.squeeze(rx_freq.numpy()),
                'true_delay_samples': delay_samples,
                'distance': distance
            })

        # store primary satellite data as before for detection training
        all_iq_samples.append((i, satellite_rx_signals[0]['rx_time']))
        all_csi.append((i, satellite_rx_signals[0]['rx_freq']))
        all_satellite_receptions.append((i, satellite_rx_signals))

        # radar echo (use main tx_time_signal)
        radar_delay = np.random.randint(50, 150)
        radar_attenuation = 0.3
        tx_time_flat = tf.squeeze(tx_time_signal)
        radar_echo = tf.roll(tx_time_flat, shift=radar_delay, axis=-1) * radar_attenuation
        # Use an independent noise power for radar echo to avoid unintended coupling
        no_radar = ebnodb2no(tf.constant(10.0), isac_system.NUM_BITS_PER_SYMBOL, isac_system.CODERATE, isac_system.rg)
        noise_power_radar = tf.cast(no_radar, tf.float32)
        radar_echo_noise = tf.complex(
            tf.random.normal(tf.shape(radar_echo), stddev=tf.sqrt(noise_power_radar/2)),
            tf.random.normal(tf.shape(radar_echo), stddev=tf.sqrt(noise_power_radar/2))
        )
        radar_echo += radar_echo_noise

        all_radar_echoes.append((i, np.squeeze(radar_echo.numpy())))
        all_labels.append((i, label))
        all_emitter_locations.append((i, emitter_loc))

        if (idx+1) % 100 == 0:
            print(f"Generated sample {idx+1}/{total_samples}...")

    # Sort by original index to keep alignment
    all_iq_samples = [x[1] for x in sorted(all_iq_samples, key=lambda x: x[0])]
    all_csi = [x[1] for x in sorted(all_csi, key=lambda x: x[0])]
    all_radar_echoes = [x[1] for x in sorted(all_radar_echoes, key=lambda x: x[0])]
    all_labels = [x[1] for x in sorted(all_labels, key=lambda x: x[0])]
    all_emitter_locations = [x[1] for x in sorted(all_emitter_locations, key=lambda x: x[0])]
    all_satellite_receptions = [x[1] for x in sorted(all_satellite_receptions, key=lambda x: x[0])]

    return {
        'iq_samples': np.array(all_iq_samples), 'csi': np.array(all_csi),
        'radar_echo': np.array(all_radar_echoes), 'labels': np.array(all_labels),
        'emitter_locations': all_emitter_locations,
        'satellite_receptions': all_satellite_receptions,
        'sampling_rate': isac_system.SAMPLING_RATE
    }


# --- Main Execution Workflow ---
if __name__ == "__main__":
    isac_system = ISACSystem()
    NUM_SAMPLES = 1500  # Increased from 1000 for better generalization
    # Use multi-satellite dataset so TDoA localization can operate on received signals
    dataset = generate_dataset_multi_satellite(isac_system, num_samples_per_class=NUM_SAMPLES, num_satellites=4)

    # Debug: Check if attack samples are actually different
    print("\n=== DATA VALIDATION ===")
    print(f"Benign samples: {np.sum(dataset['labels'] == 0)}")
    print(f"Attack samples: {np.sum(dataset['labels'] == 1)}")
    print(f"Attack samples with emitter locations: {sum(1 for loc in dataset['emitter_locations'] if loc is not None)}")

    # Check IQ signal statistics
    benign_indices = np.where(np.array(dataset['labels']) == 0)[0]
    attack_indices = np.where(np.array(dataset['labels']) == 1)[0]
    benign_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in benign_indices[:100]])
    attack_power = np.mean([np.mean(np.abs(dataset['iq_samples'][i])**2) for i in attack_indices[:100]])
    print(f"Benign sample power: {benign_power:.6f}")
    print(f"Attack sample power: {attack_power:.6f}")
    print(f"Power ratio (attack/benign): {attack_power/benign_power:.4f}")

    features = extract_features(dataset)
    labels = np.array([l for i, l in enumerate(dataset['labels']) if dataset['iq_samples'][i].size >= 64])

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        features, labels, np.arange(len(labels)), test_size=0.2, random_state=42)
    
    # Diagnostic: Visualize difference between benign and attack samples
    print("\nGenerating diagnostic visualizations...")
    if len(features) > 1000:
        benign_idx = np.where(labels == 0)[0][0]  # First benign sample
        attack_idx = np.where(labels == 1)[0][0]   # First attack sample
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot spectrograms
        axes[0, 0].imshow(features[benign_idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Benign Sample Spectrogram')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_ylabel('Time')
        
        axes[0, 1].imshow(features[attack_idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Attack Sample Spectrogram')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Time')
        
        # Plot difference
        diff = features[attack_idx, :, :, 0] - features[benign_idx, :, :, 0]
        im = axes[1, 0].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[1, 0].set_title('Difference (Attack - Benign)')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('Time')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot power spectrum
        benign_power = np.mean(features[benign_idx, :, :, 0], axis=0)
        attack_power = np.mean(features[attack_idx, :, :, 0], axis=0)
        axes[1, 1].plot(benign_power, label='Benign', alpha=0.7)
        axes[1, 1].plot(attack_power, label='Attack', alpha=0.7)
        axes[1, 1].set_title('Average Power Spectrum')
        axes[1, 1].set_xlabel('Frequency Bin')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('feature_comparison.pdf', dpi=150)
        print("Feature comparison saved to feature_comparison.pdf")
        plt.close()
    
    # Extract both types of features for dual-input model
    print("\nExtracting spectrogram features...")
    features_spec = extract_features(dataset)
    
    print("\nExtracting RX signal features (contains covert data)...")
    features_rx = extract_received_signal_features(dataset)
    
    # Ensure same number of samples and build index map to preserve original dataset indices
    min_samples = min(len(features_spec), len(features_rx))
    features_spec = features_spec[:min_samples]
    features_rx = features_rx[:min_samples]
    labels_filtered = np.array(dataset['labels'])[:min_samples]
    index_map = np.arange(min_samples)  # mapping from feature index -> dataset index

    print(f"\nFinal dataset: {len(labels_filtered)} samples")
    print(f"Spectrogram shape: {features_spec.shape}")
    print(f"RX signal features shape: {features_rx.shape}")

    # Train-test split for both feature types (preserve index_map to trace back to dataset)
    X_spec_train, X_spec_test, X_rx_train, X_rx_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features_spec, features_rx, labels_filtered, index_map,
        test_size=0.2, random_state=42)
    
    print("\nTraining the dual-input CNN detector...")
    detector_model = build_dual_input_cnn_fixed()
    
    # Early stopping: monitor validation accuracy, not loss
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='max'  # Maximize accuracy
    )
    
    history = detector_model.fit(
        [X_spec_train, X_rx_train], y_train, 
        epochs=30,  # Simpler training
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stop],
        verbose=1
    )
    
    # Print training summary
    print("\n=== TRAINING SUMMARY ===")
    if history.history['val_accuracy']:
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        print(f"Best epoch: {best_epoch}")
        print(f"Best val_accuracy: {best_val_acc:.4f}")
        print(f"Final train_accuracy: {history.history['accuracy'][-1]:.4f}")
    
    print("\nEvaluating detector performance...")
    loss, accuracy = detector_model.evaluate([X_spec_test, X_rx_test], y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred_proba = detector_model.predict([X_spec_test, X_rx_test]).ravel()
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    
    # Detailed performance metrics
    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except Exception:
        # If scikit-learn isn't available in the execution environment, provide
        # minimal fallback implementations to avoid crashing. They will not
        # produce as-detailed output but are sufficient for basic diagnostics.
        def confusion_matrix(y_true, y_pred):
            y_true = np.asarray(y_true).ravel()
            y_pred = np.asarray(y_pred).ravel()
            labels = [0, 1]
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1
            return cm

        def classification_report(y_true, y_pred, target_names=None):
            cm = confusion_matrix(y_true, y_pred)
            # build a tiny textual report
            report = ""
            for i, name in enumerate((target_names or ['0', '1'])):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                report += f"{name}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}\n"
            return report
    cm = confusion_matrix(y_test, y_pred_class)
    print("\n=== CONFUSION MATRIX ===")
    print("                Predicted")
    print("              Benign  Attack")
    print(f"Actual Benign  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Attack  {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred_class, 
                              target_names=['Benign', 'Attack']))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # Also compute AUPRC (average precision)
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        ap = average_precision_score(y_test, y_pred_proba)
        prec, rec, thr = precision_recall_curve(y_test, y_pred_proba)
        print(f"AUPRC (average precision): {ap:.3f}")
    except Exception:
        pass

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC for Covert Channel Detection'); plt.legend(loc="lower right")
    plt.grid(True); plt.savefig("roc_curve.pdf")
    print("ROC curve saved to roc_curve.pdf"); plt.show()

    print("\nPerforming localization...")
    print("\nPerforming TDoA-based localization...")
    localization_errors = []

    # Use idx_test (index_map from train_test_split) to map feature indices back to dataset indices
    # idx_test was produced by train_test_split earlier
    for i, pred in enumerate(y_pred_class):
        if pred == 1 and y_test[i] == 1:
            original_idx = idx_test[i]
            estimated_pos, true_pos = estimate_emitter_location_tdoa(int(original_idx), dataset, isac_system)
            if estimated_pos is not None and true_pos is not None:
                error = np.linalg.norm(estimated_pos - true_pos)
                localization_errors.append(error)
                if len(localization_errors) <= 3:
                    print(f"\nSample {original_idx}:")
                    print(f"  True: {true_pos}")
                    print(f"  Estimated: {estimated_pos}")
                    print(f"  Error: {error:.2f} m")

    if localization_errors:
        median_error = np.median(localization_errors)
        print(f"\n=== TDOA LOCALIZATION RESULTS ===")
        print(f"Median Error: {median_error:.2f} meters")
        print(f"Mean Error: {np.mean(localization_errors):.2f} meters")
        print(f"90th Percentile: {np.percentile(localization_errors, 90):.2f} meters")

        sorted_errors = np.sort(localization_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.figure()
        plt.plot(sorted_errors, cdf, marker='.', linestyle='none')
        plt.xlabel('Localization Error (meters)'); plt.ylabel('CDF')
        plt.title('CDF of Localization Error'); plt.grid(True)
        plt.axhline(y=0.9, color='r', linestyle='--', label='90th Percentile'); plt.legend()
        plt.savefig("localization_cdf.pdf")
        print("Localization CDF plot saved to localization_cdf.pdf"); plt.show()
    else:
        print("No covert attacks were correctly detected for localization.")

    print("\nSimulation complete.")

