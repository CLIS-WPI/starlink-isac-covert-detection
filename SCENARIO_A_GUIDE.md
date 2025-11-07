# Scenario A — Insider@Satellite (Downlink) — Execution Guide

## 📋 Prerequisites

✅ Ensure that `INSIDER_MODE = 'sat'` in `config/settings.py`.

```python
# config/settings.py
INSIDER_MODE = 'sat'  # ✅ For Scenario A
```

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
- Results in `result/scenario_a/detection_results_cnn.json`
- Model in `model/scenario_a/cnn_detector.keras`

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
- Results in `result/scenario_a/detection_results_cnn_csi.json`
- Model in `model/scenario_a/cnn_detector_csi.keras`

**Approximate time:** ~3-5 minutes

---

### Step 5: Review Results

```bash
# View CNN-only results
cat result/scenario_a/detection_results_cnn.json | jq '.metrics'

# View CNN+CSI results
cat result/scenario_a/detection_results_cnn_csi.json | jq '.metrics'

# View meta log (per-sample metadata)
head result/scenario_a/run_meta_log.csv
head result/scenario_a/run_meta_log_csi.csv
```

---

## 📊 Expected Results

Based on previous execution with `COVERT_AMP=0.5` and `POWER_PRESERVING_COVERT=True`:

### CNN-only:
- **AUC:** ~0.9997 ✅
- **Precision:** ~1.0000
- **Recall:** ~0.4000
- **F1 Score:** ~0.5714

### CNN+CSI:
- **AUC:** ~0.9814 ✅
- **Precision:** ~0.5379
- **Recall:** ~0.9933
- **F1 Score:** ~0.6979

### Physical Metrics:
- **Power diff:** ~0.14% (ultra-covert) ✅
- **Doppler:** ~-4920 Hz (mean), ±395516 Hz (std)
- **CSI variance:** ~1.64e-02

---

## 📁 Output File Structure

```
result/scenario_a/
├── detection_results_cnn.json      # CNN-only results
├── detection_results_cnn_csi.json   # CNN+CSI results
├── run_meta_log.csv                 # Meta log CNN-only
└── run_meta_log_csi.csv             # Meta log CNN+CSI

model/scenario_a/
├── cnn_detector.keras               # CNN-only model
└── cnn_detector_csi.keras           # CNN+CSI model
```

---

## 🔄 Moving Old Files (if needed)

If you have old files in `result/`:

```bash
python3 organize_results.py
```

This script moves `result/*_sat.*` files to `result/scenario_a/`.

---

## ⚠️ Important Notes

1. **Normalization:** mean/std computed only from train data (no data leakage) ✅
2. **Injection Location:** Subcarriers 24-39 (middle band) ✅
3. **Power Preserving:** `POWER_PRESERVING_COVERT = True` ✅
4. **CSI:** Real/imag channels (dual-channel) ✅

---

## 🐛 Troubleshooting

If AUC is low (< 0.70):

1. Check that `INSIDER_MODE = 'sat'`
2. Check that `COVERT_AMP = 0.5`
3. Check that `POWER_PRESERVING_COVERT = True`
4. Rebuild the dataset
5. Run `verify_injection_correctness.py`

---

## ✅ Ready for Paper

After successful execution:
- ✅ Results stored in `result/scenario_a/`
- ✅ Models stored in `model/scenario_a/`
- ✅ Completely separate from Scenario B
- ✅ Ready for use in paper
