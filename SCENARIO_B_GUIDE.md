# Scenario B — Dual-hop Uplink-Relay-Downlink (Insider@Ground) — Execution Guide

## 📋 Overview

Scenario B represents a **dual-hop communication** scenario where:
- **Insider location**: Ground station
- **Path**: Uplink → Relay → Downlink
- **Challenge**: Significant signal attenuation and channel distortion
- **Solution**: MMSE Equalization (EQ) to recover the covert pattern

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
python3 run_scenario_b.py
```

This will:
1. Generate dataset (5000 samples with EQ)
2. Train CNN model (100 epochs)
3. Generate final report

**Time**: ~1-2 hours

### Option 2: Manual Steps

#### Step 1: Generate Dataset

```bash
python3 generate_dataset_parallel.py --scenario ground --total-samples 5000
```

**Expected output:**
- `dataset/dataset_scenario_b_5000.pkl`
- Size: ~100-120 MB
- Contains equalized `rx_grids` (post-channel with MMSE EQ)

#### Step 2: Train Model

```bash
python3 main_detection_cnn.py --scenario ground --epochs 100 --batch-size 512
```

**Expected output:**
- `model/scenario_b/cnn_detector.keras`
- `result/scenario_b/detection_results_cnn.json`

## 📊 Expected Results

### Dataset Metrics
- **Total samples**: 5000 (2500 benign, 2500 attack)
- **EQ Performance**:
  - Mean SNR Improvement: ~30-40 dB
  - Median Pattern Preservation: ~0.48-0.52
- **Power Difference**: < 0.1% (Ultra-covert)

### Detection Performance
- **AUC**: ≥ 0.90 (typically ~0.95-0.99)
- **Precision**: ≥ 0.85
- **Recall**: ≥ 0.90
- **F1 Score**: ≥ 0.87

## 🔍 Key Differences from Scenario A

| Feature | Scenario A | Scenario B |
|---------|-----------|-----------|
| **Hop count** | Single-hop | Dual-hop |
| **Insider location** | Satellite | Ground |
| **Channel** | Direct downlink | Uplink + Relay + Downlink |
| **EQ required** | No | Yes (MMSE) |
| **SNR** | Higher | Lower (needs EQ) |
| **Expected AUC** | ~0.70 | ~0.95 |

## 📁 Output Files

After running the pipeline:

```
dataset/
  └── dataset_scenario_b_5000.pkl          # Generated dataset

model/scenario_b/
  ├── cnn_detector.keras                    # Trained model
  └── cnn_detector_norm.pkl                # Normalization stats

result/scenario_b/
  ├── detection_results_cnn.json          # Results (metrics, config)
  └── run_meta_log.csv                     # Training metadata
```

## 🔧 Troubleshooting

### Issue: Dataset generation takes too long
- **Cause**: EQ processing is computationally intensive
- **Solution**: Reduce samples for testing (e.g., `--total-samples 1000`)

### Issue: Low AUC (< 0.80)
- **Check**: Verify EQ performance in metadata
  - `eq_snr_improvement_db` should be ≥ 30 dB
  - `eq_pattern_preservation` should be ≥ 0.45
- **Solution**: Regenerate dataset or check EQ parameters

### Issue: Out of memory
- **Solution**: Reduce batch size: `--batch-size 256`

## 📝 Notes

- **EQ is critical**: Scenario B requires MMSE equalization to recover the pattern
- **Dataset size**: Larger than Scenario A due to CSI storage (`complex64`)
- **Training time**: Longer due to more complex channel conditions

## 🎯 Next Steps

After successful execution:
1. Review results in `result/scenario_b/detection_results_cnn.json`
2. Compare with Scenario A results
3. Generate paper figures (ROC curves, etc.)
