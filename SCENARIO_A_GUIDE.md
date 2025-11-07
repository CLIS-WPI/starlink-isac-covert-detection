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
  --scenario sat \
  --total-samples 4000 \
  --snr-list="-5,0,5,10,15,20" \
  --covert-amp-list="0.1,0.3,0.5,0.7" \
  --doppler-scale-list="0.5,1.0,1.5" \
  --pattern="fixed,random" \
  --subband="mid,random16" \
  --samples-per-config 80
```

**Explanation:**
- `--scenario sat`: Scenario A (satellite insider)
- `--total-samples 4000`: 4000 total samples (2000 per class)
- `--snr-list`: SNR range from -5 to 20 dB
- `--covert-amp-list`: Covert amplitude range (0.1 to 0.7)
- `--doppler-scale-list`: Doppler scale factors
- `--pattern`: Fixed or random patterns
- `--subband`: Middle band (24-39) or random 16 subcarriers
- `--samples-per-config`: 80 samples per configuration
- Dataset saved to `dataset/dataset_scenario_a_*.pkl` (auto-named)

**Approximate time:** ~15-20 minutes (depending on GPU)

---

### Step 2: Validate Dataset (Optional but Recommended)

```bash
# General dataset validation (auto-detects latest dataset)
python3 validate_dataset.py

# Or specify dataset explicitly
python3 validate_dataset.py --dataset dataset/dataset_scenario_a_10k.pkl
```

**Auto-Detection Feature:**
- ✅ Script automatically finds the latest `dataset_scenario_a*.pkl` file
- ✅ No need to specify dataset path manually
- ✅ If provided path doesn't exist, falls back to latest dataset

**Expected:**
- ✅ Power diff < 5%
- ✅ Pattern boost in subcarriers 24-39 (or random 16)
- ✅ Doppler non-zero and reasonable
- ✅ Labels: 50/50 split
- ✅ Normalization leakage check passed

---

### Step 3: Train CNN-only

```bash
python3 main_detection_cnn.py \
  --scenario sat \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--scenario sat`: Scenario A (satellite insider)
- `--epochs 50`: Maximum 50 epochs (with early stopping)
- `--batch-size 512`: Optimized for H100 GPU
- **Auto-detects latest dataset**: Script finds latest `dataset_scenario_a*.pkl`
- Results in `result/scenario_a/detection_results_cnn.json`
- Model in `model/scenario_a/cnn_detector.keras`

**Approximate time:** ~2-3 minutes

---

### Step 4: Train CNN+CSI (Optional)

```bash
python3 main_detection_cnn.py \
  --scenario sat \
  --use-csi \
  --epochs 50 \
  --batch-size 512
```

**Explanation:**
- `--scenario sat`: Scenario A (satellite insider)
- `--use-csi`: Enable CSI fusion (real/imag channels)
- Results in `result/scenario_a/detection_results_cnn_csi.json`
- Model in `model/scenario_a/cnn_detector_csi.keras`

**Note:** CNN-only achieves excellent results (AUC = 1.0), CSI fusion is optional for future work.

**Approximate time:** ~3-5 minutes

---

### Step 5: Run Baselines (Optional)

```bash
# Run baselines (auto-detects latest dataset)
python3 detector_baselines.py

# Or specify dataset explicitly
python3 detector_baselines.py --dataset dataset/dataset_scenario_a_10k.pkl
```

**Auto-Detection Feature:**
- ✅ Script automatically finds the latest `dataset_scenario_a*.pkl` file
- ✅ Results saved to `result/baselines_scenario_a.csv`

### Step 6: Review Results

```bash
# View CNN-only results
cat result/scenario_a/detection_results_cnn.json | jq '.metrics'

# View CNN+CSI results (if trained)
cat result/scenario_a/detection_results_cnn_csi.json | jq '.metrics'

# View baselines
cat result/baselines_scenario_a.csv

# View meta log (per-sample metadata)
head result/scenario_a/run_meta_log.csv
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

## 🚀 Complete Pipeline (Automated)

For automated execution of all steps:

```bash
# Run complete pipeline for Scenario A
./run_complete_pipeline.sh
```

This script:
1. Generates dataset for Scenario A
2. Validates dataset (auto-detects latest)
3. Trains CNN (auto-detects latest dataset)
4. Runs baselines (auto-detects latest dataset)

**Note:** The script handles both Scenario A and B sequentially. To run only Scenario A, modify the script or use individual commands above.

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
