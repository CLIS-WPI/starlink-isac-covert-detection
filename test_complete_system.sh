#!/bin/bash
# ======================================
# 🔧 Complete System Test Script
# ======================================
# Tests all phases with 2000 samples
# ======================================

# Note: We don't use 'set -e' because some steps (validation, baselines) may fail non-critically
# Each critical step has its own error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_SAMPLES=2000
NUM_SATELLITES=12
SCENARIO_A="sat"
SCENARIO_B="ground"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🔧 COMPLETE SYSTEM TEST${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Samples per class: ${NUM_SAMPLES}"
echo -e "Total samples: $((NUM_SAMPLES * 2))"
echo -e "Number of satellites: ${NUM_SATELLITES}"
echo -e "${BLUE}========================================${NC}\n"

# ======================================
# Phase 0: Validation & Sanity Checks
# ======================================
echo -e "${YELLOW}📋 Phase 0: Validation & Sanity Checks${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Check if dataset exists, if not skip validation
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}✓ Validating Scenario A dataset...${NC}"
    python3 validate_dataset.py --dataset dataset/dataset_scenario_a.pkl || echo -e "${RED}⚠️  Validation failed (non-critical)${NC}"
else
    echo -e "${YELLOW}⚠️  Scenario A dataset not found, skipping validation${NC}"
fi

if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}✓ Validating Scenario B dataset...${NC}"
    python3 validate_dataset.py --dataset dataset/dataset_scenario_b.pkl || echo -e "${RED}⚠️  Validation failed (non-critical)${NC}"
else
    echo -e "${YELLOW}⚠️  Scenario B dataset not found, skipping validation${NC}"
fi

echo ""

# ======================================
# Phase 1: Dataset Generation
# ======================================
echo -e "${YELLOW}📊 Phase 1: Dataset Generation${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Update settings for Scenario A first
echo -e "${YELLOW}📝 Updating settings for Scenario A...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
from run_all_scenarios import update_settings_file
update_settings_file('INSIDER_MODE', '\"sat\"')
update_settings_file('NUM_SAMPLES_PER_CLASS', '${NUM_SAMPLES}')
update_settings_file('NUM_SATELLITES_FOR_TDOA', '${NUM_SATELLITES}')
"

# Scenario A
echo -e "${GREEN}📡 Generating Scenario A dataset (${NUM_SAMPLES} samples per class)...${NC}"
SNR_LIST="-5,0,5,10,15,20"
AMP_LIST="0.1,0.3,0.5,0.7"
DOPPLER_LIST="0.5,1.0,1.5"
PATTERN_LIST="fixed,random"
SUBBAND_LIST="mid,random16"
python3 generate_dataset_parallel.py \
    --scenario "${SCENARIO_A}" \
    --total-samples $((NUM_SAMPLES * 2)) \
    --snr-list="$SNR_LIST" \
    --covert-amp-list="$AMP_LIST" \
    --doppler-scale-list="$DOPPLER_LIST" \
    --pattern="$PATTERN_LIST" \
    --subband="$SUBBAND_LIST" \
    --samples-per-config 40 \
    --output-csv "result/dataset_metadata_phase1_scenario_a.csv" || {
    echo -e "${RED}❌ Dataset generation failed for Scenario A${NC}"
    exit 1
}

# Rename to scenario-specific name (handle both _10k and _4000 naming)
if [ -f "dataset/dataset_scenario_a_10k.pkl" ]; then
    mv "dataset/dataset_scenario_a_10k.pkl" "dataset/dataset_scenario_a.pkl"
    echo -e "${GREEN}✓ Scenario A dataset saved as dataset_scenario_a.pkl${NC}"
elif [ -f "dataset/dataset_scenario_a_$((NUM_SAMPLES * 2)).pkl" ]; then
    mv "dataset/dataset_scenario_a_$((NUM_SAMPLES * 2)).pkl" "dataset/dataset_scenario_a.pkl"
    echo -e "${GREEN}✓ Scenario A dataset saved as dataset_scenario_a.pkl${NC}"
elif [ -f "dataset/dataset_samples${NUM_SAMPLES}_sats${NUM_SATELLITES}.pkl" ]; then
    # Fallback: use default naming
    mv "dataset/dataset_samples${NUM_SAMPLES}_sats${NUM_SATELLITES}.pkl" "dataset/dataset_scenario_a.pkl"
    echo -e "${GREEN}✓ Scenario A dataset saved as dataset_scenario_a.pkl${NC}"
fi

echo ""

# Scenario B
echo -e "${GREEN}📡 Generating Scenario B dataset (${NUM_SAMPLES} samples per class)...${NC}"
python3 generate_dataset_parallel.py \
    --scenario "${SCENARIO_B}" \
    --total-samples $((NUM_SAMPLES * 2)) \
    --snr-list="$SNR_LIST" \
    --covert-amp-list="$AMP_LIST" \
    --doppler-scale-list="$DOPPLER_LIST" \
    --pattern="$PATTERN_LIST" \
    --subband="$SUBBAND_LIST" \
    --samples-per-config 40 \
    --output-csv "result/dataset_metadata_phase1_scenario_b.csv" || {
    echo -e "${RED}❌ Dataset generation failed for Scenario B${NC}"
    exit 1
}

# Rename to scenario-specific name (handle both _10k and _4000 naming)
if [ -f "dataset/dataset_scenario_b_10k.pkl" ]; then
    mv "dataset/dataset_scenario_b_10k.pkl" "dataset/dataset_scenario_b.pkl"
    echo -e "${GREEN}✓ Scenario B dataset saved as dataset_scenario_b.pkl${NC}"
elif [ -f "dataset/dataset_scenario_b_$((NUM_SAMPLES * 2)).pkl" ]; then
    mv "dataset/dataset_scenario_b_$((NUM_SAMPLES * 2)).pkl" "dataset/dataset_scenario_b.pkl"
    echo -e "${GREEN}✓ Scenario B dataset saved as dataset_scenario_b.pkl${NC}"
elif [ -f "dataset/dataset_samples${NUM_SAMPLES}_sats${NUM_SATELLITES}.pkl" ]; then
    # Fallback: use default naming
    mv "dataset/dataset_samples${NUM_SAMPLES}_sats${NUM_SATELLITES}.pkl" "dataset/dataset_scenario_b.pkl"
    echo -e "${GREEN}✓ Scenario B dataset saved as dataset_scenario_b.pkl${NC}"
fi

echo ""

# ======================================
# Phase 2: Baseline Detectors
# ======================================
echo -e "${YELLOW}📊 Phase 2: Baseline Detectors${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Scenario A baselines
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}🔍 Running baselines for Scenario A...${NC}"
    python3 detector_baselines.py \
        --dataset dataset/dataset_scenario_a.pkl \
        --skip-cyclo || echo -e "${RED}⚠️  Baselines failed (non-critical)${NC}"
fi

# Scenario B baselines
if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}🔍 Running baselines for Scenario B...${NC}"
    python3 detector_baselines.py \
        --dataset dataset/dataset_scenario_b.pkl \
        --skip-cyclo || echo -e "${RED}⚠️  Baselines failed (non-critical)${NC}"
fi

echo ""

# ======================================
# Phase 3: CNN Training (Main Models)
# ======================================
echo -e "${YELLOW}🧠 Phase 3: CNN Training${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Update settings for Scenario A
echo -e "${YELLOW}📝 Updating settings for Scenario A...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
from run_all_scenarios import update_settings_file
update_settings_file('INSIDER_MODE', '\"sat\"')
update_settings_file('NUM_SAMPLES_PER_CLASS', '${NUM_SAMPLES}')
update_settings_file('NUM_SATELLITES_FOR_TDOA', '${NUM_SATELLITES}')
"

# Scenario A: CNN-only
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}🤖 Training CNN-only for Scenario A...${NC}"
    python3 main_detection_cnn.py \
        --epochs 50 \
        --batch-size 512 || echo -e "${RED}⚠️  CNN-only training failed${NC}"
    # Note: main_detection_cnn.py auto-detects dataset based on INSIDER_MODE
fi

# Scenario A: CNN+CSI
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}🤖 Training CNN+CSI for Scenario A...${NC}"
    python3 main_detection_cnn.py \
        --use-csi \
        --epochs 50 \
        --batch-size 512 || echo -e "${RED}⚠️  CNN+CSI training failed${NC}"
fi

# Update settings for Scenario B
echo -e "${YELLOW}📝 Updating settings for Scenario B...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
from run_all_scenarios import update_settings_file
update_settings_file('INSIDER_MODE', '\"ground\"')
"

# Scenario B: CNN-only
if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}🤖 Training CNN-only for Scenario B...${NC}"
    python3 main_detection_cnn.py \
        --epochs 50 \
        --batch-size 512 || echo -e "${RED}⚠️  CNN-only training failed${NC}"
fi

# Scenario B: CNN+CSI
if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}🤖 Training CNN+CSI for Scenario B...${NC}"
    python3 main_detection_cnn.py \
        --use-csi \
        --epochs 50 \
        --batch-size 512 || echo -e "${RED}⚠️  CNN+CSI training failed${NC}"
fi

echo ""

# ======================================
# Phase 4: Cross-Validation
# ======================================
echo -e "${YELLOW}📊 Phase 4: Cross-Validation${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Scenario A CV
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}🔄 Running 5-fold CV for Scenario A (CNN-only)...${NC}"
    python3 run_cross_validation.py \
        --scenario ${SCENARIO_A} \
        --dataset dataset/dataset_scenario_a.pkl \
        --kfold 5 \
        --seeds 3 \
        --epochs 30 \
        --batch-size 512 || echo -e "${RED}⚠️  CV failed (non-critical)${NC}"
fi

# Scenario B CV
if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}🔄 Running 5-fold CV for Scenario B (CNN-only)...${NC}"
    python3 run_cross_validation.py \
        --scenario ${SCENARIO_B} \
        --dataset dataset/dataset_scenario_b.pkl \
        --kfold 5 \
        --seeds 3 \
        --epochs 30 \
        --batch-size 512 || echo -e "${RED}⚠️  CV failed (non-critical)${NC}"
fi

echo ""

# ======================================
# Phase 5: CSI Quality Analysis
# ======================================
echo -e "${YELLOW}📊 Phase 5: CSI Quality Analysis${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Scenario A CSI analysis
if [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}📈 Analyzing CSI quality for Scenario A...${NC}"
    python3 analyze_csi.py \
        --scenario ${SCENARIO_A} \
        --dataset dataset/dataset_scenario_a.pkl \
        --bins 20 || echo -e "${RED}⚠️  CSI analysis failed (non-critical)${NC}"
fi

# Scenario B CSI analysis
if [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}📈 Analyzing CSI quality for Scenario B...${NC}"
    python3 analyze_csi.py \
        --scenario ${SCENARIO_B} \
        --dataset dataset/dataset_scenario_b.pkl \
        --bins 20 || echo -e "${RED}⚠️  CSI analysis failed (non-critical)${NC}"
fi

echo ""

# ======================================
# Phase 6: Robustness Sweep
# ======================================
echo -e "${YELLOW}📊 Phase 6: Robustness Sweep${NC}"
echo -e "${BLUE}----------------------------------------${NC}\n"

# Scenario A robustness
if [ -f "model/scenario_a/cnn_detector.keras" ] && [ -f "dataset/dataset_scenario_a.pkl" ]; then
    echo -e "${GREEN}🔍 Running robustness sweep for Scenario A...${NC}"
    python3 sweep_eval.py \
        --scenario "${SCENARIO_A}" \
        --dataset dataset/dataset_scenario_a.pkl \
        --snr-list="${SNR_LIST}" \
        --amp-list "${AMP_LIST}" \
        --pattern "${PATTERN_LIST}" \
        --doppler-scale-list "${DOPPLER_LIST}" || echo -e "${RED}⚠️  Robustness sweep failed (non-critical)${NC}"
fi

# Scenario B robustness
if [ -f "model/scenario_b/cnn_detector.keras" ] && [ -f "dataset/dataset_scenario_b.pkl" ]; then
    echo -e "${GREEN}🔍 Running robustness sweep for Scenario B...${NC}"
    python3 sweep_eval.py \
        --scenario "${SCENARIO_B}" \
        --dataset dataset/dataset_scenario_b.pkl \
        --snr-list="${SNR_LIST}" \
        --amp-list "${AMP_LIST}" \
        --pattern "${PATTERN_LIST}" \
        --doppler-scale-list "${DOPPLER_LIST}" || echo -e "${RED}⚠️  Robustness sweep failed (non-critical)${NC}"
fi

echo ""

# ======================================
# Final Summary
# ======================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ COMPLETE SYSTEM TEST FINISHED!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}📁 Generated Files:${NC}"
echo -e "  Datasets:"
echo -e "    - dataset/dataset_scenario_a.pkl"
echo -e "    - dataset/dataset_scenario_b.pkl"
echo -e ""
echo -e "  Models:"
echo -e "    - model/scenario_a/cnn_detector.keras"
echo -e "    - model/scenario_a/cnn_detector_csi.keras"
echo -e "    - model/scenario_b/cnn_detector.keras"
echo -e "    - model/scenario_b/cnn_detector_csi.keras"
echo -e ""
echo -e "  Results:"
echo -e "    - result/scenario_a/"
echo -e "    - result/scenario_b/"
echo -e "    - result/baselines_scenario_a.csv"
echo -e "    - result/baselines_scenario_b.csv"
echo -e "    - result/cv_summary_scenario_a.csv"
echo -e "    - result/cv_summary_scenario_b.csv"
echo -e "    - result/csi_analysis_summary_scenario_a.csv"
echo -e "    - result/csi_analysis_summary_scenario_b.csv"
echo -e "    - result/robustness_scenario_a.csv"
echo -e "    - result/robustness_scenario_b.csv"
echo -e ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ All tests completed!${NC}"
echo -e "${GREEN}========================================${NC}\n"

