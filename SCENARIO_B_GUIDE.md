# Scenario B — Insider@Ground (Uplink → Relay → Downlink) — Execution Guide

## 📋 Prerequisites

✅ Ensure that `INSIDER_MODE = 'ground'` in `config/settings.py`.

```python
# config/settings.py
INSIDER_MODE = 'ground'  # ✅ For Scenario B
```

## 🔄 Differences from Scenario A

| Feature | Scenario A (Satellite) | Scenario B (Ground) |
|--------|------------------------|---------------------|
| **Injection Point** | Satellite downlink | Ground terminal uplink |
| **Signal Path** | Direct downlink | Uplink → Relay → Downlink |
| **Channel Effects** | Single channel | Dual-hop (uplink + downlink) |
| **Doppler** | Single Doppler | Independent Dopplers (UL/DL) |
| **Relay** | N/A | AF relay with AGC (gain 0.5-2.0) |
| **Delay** | N/A | Processing delay (3-5 samples) |
| **Equalization** | N/A | MMSE equalization (LMMSE CSI) |
| **Expected AUC** | ~1.0 (CNN-only) | ~0.99 (CNN-only with MMSE) |
| **Pattern Preservation** | N/A | 0.4-0.5 (with MMSE) |
| **SNR Improvement** | N/A | 5-15 dB (after MMSE) |

## 🚀 Execution Commands

### Step 1: Generate Dataset

```bash
python3 generate_dataset_parallel.py \
  --scenario ground \
  --total-samples 4000 \
  --snr-list="-5,0,5,10,15,20" \
  --covert-amp-list="0.1,0.3,0.5,0.7" \
  --doppler-scale-list="0.5,1.0,1.5" \
  --pattern="fixed,random" \
  --subband="mid,random16" \
  --samples-per-config 80
```

**Explanation:**
- `--scenario ground`: Scenario B (ground insider)
- `--total-samples 4000`: 4000 total samples (2000 per class)
- `--snr-list`: SNR range from -5 to 20 dB
- `--covert-amp-list`: Covert amplitude range (0.1 to 0.7)
- `--doppler-scale-list`: Doppler scale factors
- `--pattern`: Fixed or random patterns
- `--subband`: Middle band (24-39) or random 16 subcarriers
- `--samples-per-config`: 80 samples per configuration
- Dataset saved to `dataset/dataset_scenario_b_*.pkl` (auto-named)
- **MMSE Equalization**: Automatically applied during dataset generation

**Phase 6 Features:**
- ✅ Dual-hop architecture (uplink → relay → downlink)
- ✅ Independent Dopplers (`fd_ul` and `fd_dl`)
- ✅ AF relay with AGC (gain limits 0.5-2.0)
- ✅ Processing delay (3-5 samples)
- ✅ MMSE equalization with LMMSE CSI estimation
- ✅ SNR improvement tracking (5-15 dB gain)

**Approximate time:** ~20-25 minutes (depending on GPU, includes MMSE processing)

---

### Step 2: Validate Dataset (Optional but Recommended)

```bash
# General dataset validation (auto-detects latest dataset)
python3 validate_dataset.py

# Or specify dataset explicitly
python3 validate_dataset.py --dataset dataset/dataset_scenario_b_10k.pkl
```

**Auto-Detection Feature:**
- ✅ Script automatically finds the latest `dataset_scenario_b*.pkl` file
- ✅ No need to specify dataset path manually
- ✅ If provided path doesn't exist, falls back to latest dataset

**Expected:**
- ✅ Power diff < 5%
- ✅ Pattern boost in subcarriers 24-39 (or random 16)
- ✅ Doppler non-zero and reasonable (independent UL/DL)
- ✅ Labels: 50/50 split
- ✅ Insider mode: 'ground'
- ✅ Phase 6 metadata present: `fd_ul`, `fd_dl`, `G_r_mean`, `delay_samples`, `eq_snr_improvement_db`
- ✅ Normalization leakage check passed

---

### Step 3: Train CNN-only

```bash
python3 main_detection_cnn.py \
  --scenario ground \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--scenario ground`: Scenario B (ground insider)
- `--epochs 50`: Maximum 50 epochs (with early stopping)
- `--batch-size 512`: Optimized for H100 GPU
- **Auto-detects latest dataset**: Script finds latest `dataset_scenario_b*.pkl`
- **Uses equalized signals**: Dataset contains MMSE-equalized `rx_grids`
- Results in `result/scenario_b/detection_results_cnn.json`
- Model in `model/scenario_b/cnn_detector.keras`

**Note:** The CNN trains on MMSE-equalized signals, which significantly improves pattern preservation and detection performance.

**Approximate time:** ~2-3 minutes

---

### Step 4: Train CNN+CSI (Optional)

```bash
python3 main_detection_cnn.py \
  --scenario ground \
  --use-csi \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--scenario ground`: Scenario B (ground insider)
- `--use-csi`: Enable CSI fusion (real/imag channels)
- Results in `result/scenario_b/detection_results_cnn_csi.json`
- Model in `model/scenario_b/cnn_detector_csi.keras`

**Note:** CNN-only with MMSE equalization achieves excellent results (AUC = 0.9917), CSI fusion is optional for future work.

**Approximate time:** ~3-5 minutes

---

### Step 5: Run Baselines (Optional)

```bash
# Run baselines (auto-detects latest dataset)
python3 detector_baselines.py

# Or specify dataset explicitly
python3 detector_baselines.py --dataset dataset/dataset_scenario_b_10k.pkl
```

**Auto-Detection Feature:**
- ✅ Script automatically finds the latest `dataset_scenario_b*.pkl` file
- ✅ Results saved to `result/baselines_scenario_b.csv`

### Step 6: Review Results

```bash
# View CNN-only results
cat result/scenario_b/detection_results_cnn.json | jq '.metrics'

# View CNN+CSI results (if trained)
cat result/scenario_b/detection_results_cnn_csi.json | jq '.metrics'

# View baselines
cat result/baselines_scenario_b.csv

# View meta log (per-sample metadata)
head result/scenario_b/run_meta_log.csv

# Check Phase 6 metadata (SNR improvement, delay, gain)
python3 -c "import pickle; d=pickle.load(open('dataset/dataset_scenario_b_10k.pkl','rb')); print('SNR improvement:', d['scenario_b_meta'].get('eq_snr_improvement_db', 'N/A'))"
```

---

## 📊 Expected Results

Based on Phase 6 implementation with MMSE equalization:

### CNN-only (with MMSE):
- **AUC:** ~0.9917 ✅ (excellent, close to Scenario A)
- **Precision:** ~0.95+
- **Recall:** ~0.95+
- **F1 Score:** ~0.95+

### Physical Metrics:
- **Power diff:** ~0.12% (ultra-covert) ✅
- **Pattern Preservation:** 0.4-0.5 (with MMSE equalization) ✅
- **SNR Improvement:** 5-15 dB (after MMSE) ✅
- **Doppler (UL):** Independent from DL
- **Doppler (DL):** Independent from UL
- **Relay Gain:** 0.5-2.0 (AGC controlled)
- **Relay Delay:** 3-5 samples

### Technical Notes:
- **MMSE Equalization**: Significantly improves pattern preservation and detection performance
- **Dual-hop Architecture**: More challenging than Scenario A, but MMSE compensates
- **Independent Dopplers**: Realistic modeling of separate uplink and downlink channels

---

## 📁 Output File Structure

```
result/scenario_b/
├── detection_results_cnn.json      # CNN-only results
├── detection_results_cnn_csi.json   # CNN+CSI results
├── run_meta_log.csv                 # Meta log CNN-only
└── run_meta_log_csi.csv             # Meta log CNN+CSI

model/scenario_b/
├── cnn_detector.keras               # CNN-only model
└── cnn_detector_csi.keras           # CNN+CSI model
```

---

## 🔄 Comparison with Scenario A

After executing Scenario B, you can compare results:

```bash
# Compare AUC
echo "Scenario A - CNN-only:"
cat result/scenario_a/detection_results_cnn.json | jq '.metrics.auc'
echo "Scenario B - CNN-only:"
cat result/scenario_b/detection_results_cnn.json | jq '.metrics.auc'

echo "Scenario A - CNN+CSI:"
cat result/scenario_a/detection_results_cnn_csi.json | jq '.metrics.auc'
echo "Scenario B - CNN+CSI:"
cat result/scenario_b/detection_results_cnn_csi.json | jq '.metrics.auc'
```

---

## ⚠️ Important Notes

1. **Normalization:** mean/std computed only from train data (no data leakage) ✅
2. **Injection Location:** Subcarriers 24-39 (middle band) or random 16 ✅
3. **Power Preserving:** `POWER_PRESERVING_COVERT = True` ✅
4. **MMSE Equalization:** Applied during dataset generation, stored in `rx_grids` ✅
5. **Phase 6 Complete:**
   - Dual-hop architecture (uplink → relay → downlink) ✅
   - Independent Dopplers (`fd_ul` and `fd_dl`) ✅
   - AF relay with AGC (gain 0.5-2.0) ✅
   - Processing delay (3-5 samples) ✅
   - MMSE equalization with LMMSE CSI estimation ✅
6. **Auto-Detection:** Scripts automatically find latest dataset ✅

---

## 🐛 Troubleshooting

If AUC is low (< 0.90):

1. Check that `INSIDER_MODE = 'ground'`
2. Check that MMSE equalization is applied (check dataset metadata)
3. Verify Phase 6 metadata exists: `fd_ul`, `fd_dl`, `eq_snr_improvement_db`
4. Check pattern preservation (should be 0.4-0.5 with MMSE)
5. Rebuild the dataset with MMSE equalization
6. Run `validate_dataset.py` to check dataset integrity
7. Check that latest dataset is being used (auto-detection)

**Note:** With MMSE equalization, Scenario B should achieve AUC ≥ 0.99. If not, check:
- CSI estimation quality (LMMSE should be used)
- SNR improvement metrics (should show 5-15 dB gain)
- Pattern preservation (should be 0.4-0.5)

---

## 🚀 Complete Pipeline (Automated)

For automated execution of all steps:

```bash
# Run complete pipeline for Scenario B
./run_complete_pipeline.sh
```

This script:
1. Generates dataset for Scenario B (with MMSE equalization)
2. Validates dataset (auto-detects latest)
3. Trains CNN (auto-detects latest dataset)
4. Runs baselines (auto-detects latest dataset)

**Note:** The script handles both Scenario A and B sequentially. To run only Scenario B, modify the script or use individual commands above.

## ✅ Ready for Paper

After successful execution:
- ✅ Results stored in `result/scenario_b/`
- ✅ Models stored in `model/scenario_b/`
- ✅ Phase 6 metadata stored in dataset
- ✅ MMSE equalization applied and validated
- ✅ Completely separate from Scenario A
- ✅ Ready for comparison and use in paper
