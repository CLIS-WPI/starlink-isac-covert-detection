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
| **Channel Effects** | Single channel | Double channel (uplink + downlink) |
| **Noise** | Single noise | Double noise (relay amplifies noise) |
| **Expected AUC** | ~1.0 (CNN-only) | ~0.85-0.95 (CNN-only) |
| | ~0.96 (CNN+CSI) | ~0.90+ (CNN+CSI) |

## 🚀 Execution Commands

### Step 1: Generate Dataset

```bash
python3 generate_dataset_parallel.py \
  --num-samples 500 \
  --num-satellites 12
```

**Explanation:**
- `--num-samples 500`: 500 samples per class = 1000 total samples
- `--num-satellites 12`: 12 satellites for TDoA
- Dataset saved to `dataset/dataset_samples500_sats12.pkl`
- ⚠️ **Note:** If you want to keep Scenario A dataset, rename it first:

```bash
# Keep Scenario A dataset
mv dataset/dataset_samples500_sats12.pkl dataset/dataset_scenario_a.pkl

# After generating Scenario B dataset
mv dataset/dataset_samples500_sats12.pkl dataset/dataset_scenario_b.pkl
```

**Approximate time:** ~10-15 minutes (depending on GPU)

---

### Step 2: Validate Dataset (Optional but Recommended)

```bash
# General dataset validation
python3 validate_dataset.py

# Check injection correctness (pre-channel, power_diff_pct, pattern_boost, doppler_hz)
python3 verify_injection_correctness.py

# Check consistency (for multi-GPU)
python3 check_dataset_consistency.py
```

**Expected:**
- ✅ Power diff < 5%
- ✅ Pattern boost in subcarriers 24-39
- ✅ Doppler non-zero and reasonable
- ✅ Labels: 50/50 split
- ✅ Insider mode: 'ground'

---

### Step 3: Train CNN-only

```bash
python3 main_detection_cnn.py \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--epochs 50`: Maximum 50 epochs (with early stopping)
- `--batch-size 512`: Optimized for H100 GPU
- Results in `result/scenario_b/detection_results_cnn.json`
- Model in `model/scenario_b/cnn_detector.keras`

**Approximate time:** ~2-3 minutes

---

### Step 4: Train CNN+CSI

```bash
python3 main_detection_cnn.py \
  --use-csi \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--use-csi`: Enable CSI fusion (real/imag channels)
- Results in `result/scenario_b/detection_results_cnn_csi.json`
- Model in `model/scenario_b/cnn_detector_csi.keras`

**Approximate time:** ~3-5 minutes

---

### Step 5: Review Results

```bash
# View CNN-only results
cat result/scenario_b/detection_results_cnn.json | jq '.metrics'

# View CNN+CSI results
cat result/scenario_b/detection_results_cnn_csi.json | jq '.metrics'

# View meta log (per-sample metadata)
head result/scenario_b/run_meta_log.csv
head result/scenario_b/run_meta_log_csi.csv
```

---

## 📊 Expected Results

Based on differences between Scenario B and A:

### CNN-only:
- **AUC:** ~0.85-0.95 (lower than Scenario A due to relay)
- **Precision:** ~0.70-0.90
- **Recall:** ~0.30-0.50
- **F1 Score:** ~0.40-0.60

### CNN+CSI:
- **AUC:** ~0.90+ ✅ (target: ≥ 0.90)
- **Precision:** ~0.60-0.80
- **Recall:** ~0.90-0.99
- **F1 Score:** ~0.70-0.85

### Physical Metrics:
- **Power diff:** < 5% (ultra-covert) ✅
- **Doppler:** Similar to Scenario A
- **CSI variance:** May be slightly higher (due to relay)

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
2. **Injection Location:** Subcarriers 24-39 (middle band) ✅
3. **Power Preserving:** `POWER_PRESERVING_COVERT = True` ✅
4. **CSI:** Real/imag channels (dual-channel) ✅
5. **Relay Effect:** Amplify-and-Forward (double noise) ⚠️

---

## 🐛 Troubleshooting

If AUC is low (< 0.85):

1. Check that `INSIDER_MODE = 'ground'`
2. Check that `COVERT_AMP = 0.5`
3. Check that `POWER_PRESERVING_COVERT = True`
4. Rebuild the dataset
5. Run `verify_injection_correctness.py`
6. Note: Scenario B naturally has lower AUC (due to relay)

---

## ✅ Ready for Paper

After successful execution:
- ✅ Results stored in `result/scenario_b/`
- ✅ Models stored in `model/scenario_b/`
- ✅ Completely separate from Scenario A
- ✅ Ready for comparison and use in paper
