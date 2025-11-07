# 🔍 Complete Pipeline Validation Guide

## Overview

This guide describes the comprehensive validation plan for the **Real-Time Covert Leakage Detection in Large-Scale LEO Satellite Networks** pipeline.

## Quick Start

```bash
# Run dataset validation (auto-detects latest dataset)
python3 validate_dataset.py

# Or specify dataset explicitly
python3 validate_dataset.py --dataset dataset/dataset_scenario_a_10k.pkl

# Run complete pipeline validation
python3 validate_complete_pipeline.py
```

**Auto-Detection Feature:**
- ✅ `validate_dataset.py` automatically finds latest scenario-specific dataset
- ✅ `detector_baselines.py` automatically finds latest scenario-specific dataset
- ✅ No need to specify dataset path manually

## Validation Phases

### Phase 1: Structural Validation (Pipeline Consistency)

**Goal**: Ensure all required files and directories exist.

**Checks**:
- ✅ Latest `dataset_scenario_a*.pkl` exists (auto-detected)
- ✅ Latest `dataset_scenario_b*.pkl` exists (auto-detected)
- ✅ `model/scenario_a/cnn_detector.keras` exists
- ✅ `model/scenario_b/cnn_detector.keras` exists
- ✅ `result/scenario_a/detection_results_cnn.json` exists
- ✅ `result/scenario_b/detection_results_cnn.json` exists
- ✅ `result/baselines_scenario_a.csv` exists (optional)
- ✅ `result/baselines_scenario_b.csv` exists (optional)

**Optional Files** (nice to have):
- `model/scenario_a/cnn_detector_csi.keras`
- `model/scenario_b/cnn_detector_csi.keras`
- `result/baselines_scenario_a.csv`
- `result/baselines_scenario_b.csv`

---

### Phase 2: Dataset Integrity Check

**Goal**: Verify dataset correctness according to paper specifications.

**Checks**:
1. **Sample Balance**: 50/50 split (benign/attack)
2. **Auto-Detection**: Script automatically finds latest dataset ✅
3. **Phase 6 Metadata** (Scenario B):
   - `fd_ul`: Uplink Doppler shift (independent from DL)
   - `fd_dl`: Downlink Doppler shift (independent from UL)
   - `G_r_mean`: Relay gain (should be 0.5-2.0)
   - `delay_samples`: Relay processing delay (should be 3-5)
   - `eq_snr_raw_db`: Pre-equalization SNR
   - `eq_snr_improvement_db`: SNR improvement after MMSE (should be 5-15 dB)
   - `alpha_units`: Regularization parameter units ('power')
   - `alpha_is_per_subcarrier`: False (global alpha)
4. **Power Difference**:
   - Scenario A: ≈ 0.04% (threshold: ≤ 0.2%)
   - Scenario B: ≈ 0.12% (threshold: ≤ 0.2%)
5. **Pattern Preservation** (Scenario B):
   - Should be 0.4-0.5 with MMSE equalization ✅

**Expected Results**:
- Dataset A: 4000 samples (2000 benign + 2000 attack) with diverse configurations
- Dataset B: 4000 samples (2000 benign + 2000 attack) with MMSE equalization

---

### Phase 3: Model Behavior Validation

**Goal**: Verify model inference and performance match paper results.

**Checks**:
- Model loads without errors
- Inference runs successfully
- Performance metrics match paper expectations

**Expected Results** (from latest implementation):
- **Scenario A**: AUC ≈ 1.0000, F1 ≈ 0.9967
- **Scenario B**: AUC ≈ 0.9917, F1 ≈ 0.95+ (with MMSE equalization)

**Tolerance**:
- AUC: ≥ 0.99 (acceptable)
- F1: ≥ 0.95 (acceptable)
- Scenario B: Pattern preservation ≥ 0.4 (with MMSE)
- Scenario B: SNR improvement ≥ 5 dB (after MMSE)

---

### Phase 4: Scenario-Specific Validation

**Goal**: Ensure Scenario B implements full dual-hop architecture.

**Checks**:
1. **Metadata CSV** (`dataset_metadata_phase1_scenario_b.csv`):
   - Contains Phase 6 columns
   - `fd_ul` and `fd_dl` are independent (different values)
   - `eq_snr_improvement_db` present (SNR gain after MMSE)
   - `eq_snr_raw_db` present (pre-equalization SNR)
2. **Relay Function**:
   - `amplify_and_forward_relay()` exists in `core/scenario_b_relay.py`
   - Function includes AGC (`target_power` parameter, gain limits 0.5-2.0)
   - Function includes delay handling (3-5 samples)
   - Function includes clipping protection
3. **MMSE Equalization**:
   - `mmse_equalize()` exists in `core/csi_estimation.py`
   - Uses LMMSE CSI estimation
   - Adaptive regularization (α) based on SNR
   - SNR-based blending for low SNR conditions

**Expected**:
- `fd_ul` ≠ `fd_dl` (independent Doppler shifts)
- Relay gain `G_r_mean` between 0.5 and 2.0
- Delay samples between 3 and 5
- SNR improvement ≥ 5 dB (after MMSE)
- Pattern preservation ≥ 0.4 (with MMSE)

---

### Phase 5: Statistical Validation

**Goal**: Compare statistical results with paper.

**Checks**:
- Cross-validation summary statistics
- Comparison with paper expectations

**Expected Results** (from paper):
- **Scenario A**: AUC = 0.9998 ± 0.0001, F1 = 0.9933 ± 0.002
- **Scenario B**: AUC = 0.9999 ± 0.0001, F1 = 0.9740 ± 0.005

**Tolerance**:
- AUC difference: ≤ ±0.005
- F1 difference: ≤ ±0.01

---

### Phase 6: Robustness & Sanity Test

**Goal**: Verify model robustness across different conditions.

**Checks**:
- Robustness sweep results
- Minimum AUC across all conditions

**Expected**:
- Minimum AUC ≥ 0.98 (robustness threshold)
- Mean AUC should remain high across SNR/amplitude/pattern variations

**Test Command** (optional manual test):
```bash
python3 sweep_eval.py --scenario ground --amp-list "0.1,0.3,0.5,0.7"
```

---

### Phase 7: Final Acceptance Criteria

**Goal**: Final checklist for delivery readiness.

**Criteria**:
1. ✅ Pipeline executes without errors
2. ✅ Output files are complete
3. ✅ Phase 6 metadata exists in Scenario B (including MMSE metrics)
4. ✅ AUC ≥ 0.99 for both scenarios
5. ✅ Power diff ≤ 0.2% for both scenarios
6. ✅ Model inference works without crashes
7. ✅ Auto-detection works for validation and baselines
8. ✅ Scenario B: Pattern preservation ≥ 0.4 (with MMSE)
9. ✅ Scenario B: SNR improvement ≥ 5 dB (after MMSE)

**Result**:
- **ALL PASS**: Pipeline ready for final delivery ✅
- **ANY FAIL**: Review validation results and fix issues ⚠️

---

## Output Format

The validation script provides:
- ✅ **Green checkmarks**: Passed checks
- ❌ **Red X marks**: Failed checks
- ⚠️ **Yellow warnings**: Non-critical issues
- ℹ️ **Blue info**: Informational messages

## Example Output

```
======================================================================
Phase 1: Structural Validation (Pipeline Consistency)
======================================================================

━━━ Required Files ━━━
✅ Dataset A: dataset/dataset_scenario_a.pkl (45.23 MB)
✅ Dataset B: dataset/dataset_scenario_b.pkl (45.67 MB)
✅ Model A (CNN-only): model/scenario_a/cnn_detector.keras (12.34 MB)
...

━━━ Optional Files ━━━
ℹ️  Model A (CNN+CSI): model/scenario_a/cnn_detector_csi.keras (15.67 MB)
...

======================================================================
Phase 2: Dataset Integrity Check
======================================================================

━━━ Scenario A Dataset ━━━
ℹ️  Total samples: 4000
ℹ️  Benign samples: 2000 (50.0%)
ℹ️  Attack samples: 2000 (50.0%)
✅ Dataset is balanced (ratio: 1.000)
ℹ️  Mean power diff: 0.0412%
✅ Power diff within threshold (0.0412% <= 0.2%)

...
```

## Troubleshooting

### Common Issues

1. **Missing Files**:
   - Run `python3 run_all_scenarios.py` to generate datasets and models
   - Check that all phases (0-6) completed successfully

2. **Low AUC**:
   - Regenerate dataset with correct parameters
   - Check that `COVERT_AMP` is set appropriately
   - Verify normalization is applied correctly

3. **Missing Phase 6 Metadata**:
   - Ensure `INSIDER_MODE = 'ground'` for Scenario B
   - Regenerate Scenario B dataset after Phase 6 implementation
   - Check for `eq_snr_improvement_db` in dataset metadata
   - Verify MMSE equalization is applied (check `rx_grids` magnitude)

4. **Power Diff Too High**:
   - Check `POWER_PRESERVING_COVERT = True` in settings
   - Verify injection logic in `covert_injection.py`

## References

- Paper: "Real-Time Covert Leakage Detection in Large-Scale LEO Satellite Networks"
- Table I: Expected performance metrics
- Section VI: Robustness analysis
- Phase 6: Dual-hop relay implementation

---

**Last Updated**: After Phase 6 completion with MMSE equalization
**Status**: ✅ Ready for validation

## 🔧 Recent Updates

1. **Auto-Detection**: Scripts now automatically find latest dataset files
2. **MMSE Equalization**: Scenario B includes MMSE equalization with SNR improvement tracking
3. **Phase 6 Complete**: Dual-hop architecture with independent Dopplers, AF relay, and MMSE
4. **Enhanced Validation**: Checks for pattern preservation and SNR improvement metrics
