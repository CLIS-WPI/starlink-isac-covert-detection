# 📊 Results Summary — Real-Time Covert Leakage Detection

## 🎯 Final Decision for Paper

**Using CNN-only for both Scenarios:**
- ✅ **CNN-only** works excellently (AUC ≥ 0.99)
- ✅ **Scenario B Phase 6 Complete**: Dual-hop with MMSE equalization
- ✅ **CSI fusion** → Future Work (due to noisy CSI)

---

## 📈 Scenario A — Insider@Satellite (Downlink)

### CNN-only (for paper):
- **AUC:** 1.0000 ✅
- **Precision:** 1.0000
- **Recall:** 0.9933 (with threshold optimization)
- **F1 Score:** 0.9967

### Physical Metrics:
- **Power diff:** 0.01% (ultra-covert) ✅
- **Doppler:** -4920.91 Hz (mean), ±395516 Hz (std)
- **Threshold (optimized):** 0.51

### Output Files:
- `result/scenario_a/detection_results_cnn.json`
- `model/scenario_a/cnn_detector.keras`

---

## 📈 Scenario B — Insider@Ground (Uplink → Relay → Downlink)

### CNN-only (for paper):
- **AUC:** 0.9917 ✅
- **Precision:** 0.95+
- **Recall:** 0.95+
- **F1 Score:** 0.95+

### Physical Metrics:
- **Power diff:** 0.12% (ultra-covert) ✅
- **Pattern Preservation:** 0.4-0.5 (with MMSE equalization) ✅
- **SNR Improvement:** 5-15 dB (after MMSE) ✅
- **Doppler (UL):** Independent from DL
- **Doppler (DL):** Independent from UL
- **Relay Gain:** 0.5-2.0 (AGC controlled)
- **Relay Delay:** 3-5 samples

### Technical Implementation:
- ✅ **Dual-hop architecture**: Uplink → AF Relay → Downlink
- ✅ **MMSE Equalization**: LMMSE CSI estimation with adaptive regularization
- ✅ **Independent Dopplers**: `fd_ul` and `fd_dl` for each hop
- ✅ **AF Relay**: Automatic Gain Control (AGC) with clipping protection

### Output Files:
- `result/scenario_b/detection_results_cnn.json`
- `model/scenario_b/cnn_detector.keras`
- `dataset/dataset_scenario_b_*.pkl` (with Phase 6 metadata)

---

## 📊 Comparison: Scenario A vs Scenario B (CNN-only)

| Metric | Scenario A | Scenario B | Notes |
|--------|------------|------------|-------|
| **AUC** | 1.0000 | 0.9917 | Both excellent ✅ |
| **Precision** | 1.0000 | 0.95+ | Both excellent ✅ |
| **Recall** | 0.9933 | 0.95+ | Both excellent ✅ |
| **F1 Score** | 0.9967 | 0.95+ | Both excellent ✅ |
| **Power diff** | 0.01% | 0.12% | Both ultra-covert ✅ |
| **Architecture** | Direct link | Dual-hop + MMSE | B more complex |
| **Pattern Preservation** | N/A | 0.4-0.5 | With MMSE ✅ |
| **SNR Improvement** | N/A | 5-15 dB | After MMSE ✅ |

**Result:** Both Scenarios have **excellent** results! ✅
- Scenario A: Perfect detection (AUC = 1.0) with direct link
- Scenario B: Excellent detection (AUC = 0.9917) with dual-hop and MMSE equalization

---

## ✅ Key Points for Paper

1. **Excellent results in both Scenarios:**
   - Scenario A: AUC = 1.0000 (perfect detection) ✅
   - Scenario B: AUC = 0.9917 (excellent detection with MMSE) ✅
   - Both scenarios achieve high precision, recall, and F1 scores ✅

2. **Ultra-Covert Detection:**
   - Power diff < 0.2% in both Scenarios ✅
   - Pattern detection without noticeable power change ✅
   - CNN-only capable of detecting very subtle patterns ✅
   - Scenario B: Pattern preservation 0.4-0.5 with MMSE equalization ✅

3. **Technical Implementation:**
   - Scenario A: Direct downlink (simpler architecture)
   - Scenario B: **Phase 6 Complete** - Dual-hop with MMSE equalization ✅
     - Independent Dopplers for uplink and downlink
     - AF relay with AGC (gain 0.5-2.0) and processing delay
     - MMSE equalization with SNR improvement (5-15 dB)
     - Pattern preservation significantly improved

4. **Pipeline Robustness:**
   - Auto-detect latest dataset in validation and baselines ✅
   - Complete pipeline script (`run_complete_pipeline.sh`) ✅
   - Parallel dataset generation ✅

5. **Future Work:**
   - **CSI Fusion:** Needs improved CSI estimation (NMSE < -10 dB)
   - **Robustness Tests:** Sweep COVERT_AMP and band position
   - **Cross-validation:** Extended validation across different conditions

---

## 📁 Output File Structure

```
result/
├── scenario_a/
│   ├── detection_results_cnn.json
│   ├── detection_results_cnn_csi.json
│   ├── run_meta_log.csv
│   └── run_meta_log_csi.csv
│
└── scenario_b/
    ├── detection_results_cnn.json
    ├── detection_results_cnn_csi.json
    ├── run_meta_log.csv
    └── run_meta_log_csi.csv

model/
├── scenario_a/
│   ├── cnn_detector.keras
│   └── cnn_detector_csi.keras
│
└── scenario_b/
    ├── cnn_detector.keras
    └── cnn_detector_csi.keras
```

---

## 🎯 Ready for Paper

✅ Results stored in separate folders (`scenario_a/` and `scenario_b/`)
✅ Models stored in separate folders
✅ Power diff < 5% in both scenarios (ultra-covert)
✅ AUC ≥ 0.95 in all cases
✅ Reproducible and documented results

---

## 📝 Settings Used

- **Detector:** CNN-only (for paper)
- **COVERT_AMP:** 0.1, 0.3, 0.5, 0.7 (diverse range)
- **SNR Range:** -5, 0, 5, 10, 15, 20 dB
- **POWER_PRESERVING_COVERT:** True
- **Injection Location:** Subcarriers 24-39 (middle band) or random 16
- **Pattern Types:** Fixed or random
- **Normalization:** Global z-score (no data leakage)
- **Threshold:** Optimized (F1-max on validation set)
- **Dataset:** 4000 samples (2000 per class) with diverse configurations
- **Scenario B:** MMSE equalization enabled, dual-hop with AF relay

## 🔮 Future Work (for paper)

1. **CSI Fusion Enhancement:**
   - Improve CSI estimation (target: NMSE < -10 dB)
   - Better smoothing and interpolation methods
   - Attention-based fusion with quality gating

2. **Robustness Analysis:**
   - Sweep COVERT_AMP (0.1 → 0.5)
   - Band position sensitivity
   - Channel condition variations
   - Cross-validation across different SNR ranges

3. **Advanced Equalization:**
   - Adaptive MMSE parameters based on channel conditions
   - Multi-tap equalization for severe channel effects
