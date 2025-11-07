#!/bin/bash
# اجرای کامل Pipeline برای Scenario A و B

set -e  # Exit on error

echo "======================================================================"
echo "🚀 اجرای کامل Pipeline: Scenario A → Scenario B"
echo "======================================================================"

# تنظیمات
NUM_SAMPLES=2000  # 2000 samples per class = 4000 total
EPOCHS=50

# ============================================================================
# SCENARIO A: Satellite Direct (insider@satellite)
# ============================================================================
echo ""
echo "======================================================================"
echo "📡 SCENARIO A: Satellite Direct"
echo "======================================================================"

# Step 1: Generate Dataset
echo ""
echo "[1/4] Generating Scenario A dataset..."
python3 generate_dataset_parallel.py \
    --scenario sat \
    --total-samples $((NUM_SAMPLES * 2)) \
    --snr-list="-5,0,5,10,15,20" \
    --covert-amp-list="0.1,0.3,0.5,0.7" \
    --doppler-scale-list="0.5,1.0,1.5" \
    --pattern="fixed,random" \
    --subband="mid,random16" \
    --samples-per-config 80

# Step 2: Validate Dataset
echo ""
echo "[2/4] Validating Scenario A dataset..."
# Find latest Scenario A dataset
SCENARIO_A_DATASET=$(ls -t dataset/dataset_scenario_a*.pkl 2>/dev/null | head -1)
if [ -z "$SCENARIO_A_DATASET" ]; then
    SCENARIO_A_DATASET="dataset/dataset_scenario_a.pkl"
fi
echo "  Using dataset: $SCENARIO_A_DATASET"
python3 validate_dataset.py --dataset "$SCENARIO_A_DATASET"

# Step 3: Train CNN
echo ""
echo "[3/4] Training CNN for Scenario A..."
python3 main_detection_cnn.py \
    --scenario sat \
    --epochs $EPOCHS \
    --batch-size 512

# Step 4: Baselines
echo ""
echo "[4/4] Running baselines for Scenario A..."
# Find latest Scenario A dataset
SCENARIO_A_DATASET=$(ls -t dataset/dataset_scenario_a*.pkl 2>/dev/null | head -1)
if [ -z "$SCENARIO_A_DATASET" ]; then
    SCENARIO_A_DATASET="dataset/dataset_scenario_a.pkl"
fi
python3 detector_baselines.py --dataset "$SCENARIO_A_DATASET"

echo ""
echo "======================================================================"
echo "✅ SCENARIO A Complete!"
echo "======================================================================"

# ============================================================================
# SCENARIO B: Dual-Hop Relay (insider@ground)
# ============================================================================
echo ""
echo "======================================================================"
echo "📡 SCENARIO B: Dual-Hop Relay"
echo "======================================================================"

# Step 1: Generate Dataset
echo ""
echo "[1/4] Generating Scenario B dataset..."
python3 generate_dataset_parallel.py \
    --scenario ground \
    --total-samples $((NUM_SAMPLES * 2)) \
    --snr-list="-5,0,5,10,15,20" \
    --covert-amp-list="0.1,0.3,0.5,0.7" \
    --doppler-scale-list="0.5,1.0,1.5" \
    --pattern="fixed,random" \
    --subband="mid,random16" \
    --samples-per-config 80

# Step 2: Validate Dataset
echo ""
echo "[2/4] Validating Scenario B dataset..."
# Find latest Scenario B dataset
SCENARIO_B_DATASET=$(ls -t dataset/dataset_scenario_b*.pkl 2>/dev/null | head -1)
if [ -z "$SCENARIO_B_DATASET" ]; then
    SCENARIO_B_DATASET="dataset/dataset_scenario_b.pkl"
fi
echo "  Using dataset: $SCENARIO_B_DATASET"
python3 validate_dataset.py --dataset "$SCENARIO_B_DATASET"

# Step 3: Train CNN
echo ""
echo "[3/4] Training CNN for Scenario B..."
python3 main_detection_cnn.py \
    --scenario ground \
    --epochs $EPOCHS \
    --batch-size 512

# Step 4: Baselines
echo ""
echo "[4/4] Running baselines for Scenario B..."
# Find latest Scenario B dataset
SCENARIO_B_DATASET=$(ls -t dataset/dataset_scenario_b*.pkl 2>/dev/null | head -1)
if [ -z "$SCENARIO_B_DATASET" ]; then
    SCENARIO_B_DATASET="dataset/dataset_scenario_b.pkl"
fi
python3 detector_baselines.py --dataset "$SCENARIO_B_DATASET"

echo ""
echo "======================================================================"
echo "✅ SCENARIO B Complete!"
echo "======================================================================"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "📊 SUMMARY"
echo "======================================================================"
echo ""
echo "Scenario A Results:"
echo "  - Dataset: dataset/dataset_scenario_a.pkl"
echo "  - Model: model/scenario_a/cnn_detector.keras"
echo "  - Results: result/scenario_a/detection_results_cnn.json"
echo ""
echo "Scenario B Results:"
echo "  - Dataset: dataset/dataset_scenario_b.pkl"
echo "  - Model: model/scenario_b/cnn_detector.keras"
echo "  - Results: result/scenario_b/detection_results_cnn.json"
echo ""
echo "======================================================================"
echo "✅ Pipeline Complete!"
echo "======================================================================"

