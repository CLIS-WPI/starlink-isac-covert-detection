#!/bin/bash
# ======================================
# 🚀 Full CNN Detection Pipeline
# ======================================
# This script:
# 1. Removes old dataset
# 2. Generates new dataset with randomization
# 3. Trains CNN detector
# ======================================

set -e  # Exit on error

echo "========================================"
echo "🚀 Full CNN Detection Pipeline"
echo "========================================"

# Step 0: Verify configuration
echo ""
echo "🔍 Step 0: Verifying configuration..."
python3 verify_config.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration errors detected! Please fix before proceeding."
    exit 1
fi

# Step 1: Remove old dataset
echo ""
echo "📦 Step 1: Removing old dataset..."
if [ -f "dataset/dataset_samples1500_sats12.pkl" ]; then
    rm -f dataset/dataset_samples1500_sats12.pkl
    echo "✓ Old dataset removed"
else
    echo "✓ No old dataset found"
fi

# Step 2: Generate new dataset with randomization
echo ""
echo "📊 Step 2: Generating new dataset..."
echo "Settings:"
echo "  - COVERT_AMP = 1.4"
echo "  - RANDOMIZE_SUBCARRIERS = True (limited to 48)"
echo "  - RANDOMIZE_SYMBOLS = True (7 symbols)"
echo "  - ADD_NOISE = True (std=0.01)"
echo ""
python3 generate_dataset_parallel.py

# Check if dataset was created successfully
if [ ! -f "dataset/dataset_samples1500_sats12.pkl" ]; then
    echo "❌ ERROR: Dataset generation failed!"
    exit 1
fi

# Analyze power before training
echo ""
echo "🔍 Step 2.5: Analyzing dataset..."
python3 analyze_power.py
python3 check_balance.py

# Step 3: Train CNN detector
echo ""
echo "🧠 Step 3: Training CNN detector..."
python3 main_detection_cnn.py

echo ""
echo "========================================"
echo "✅ Pipeline completed successfully!"
echo "========================================"
