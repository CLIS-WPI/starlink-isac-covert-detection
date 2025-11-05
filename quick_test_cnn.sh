#!/bin/bash
# ======================================
# 🚀 Quick CNN Test Pipeline
# ======================================
# For rapid iteration during hyperparameter tuning
# Uses smaller dataset (500 samples) for faster feedback
# ======================================

set -e  # Exit on error

echo "========================================"
echo "🚀 Quick CNN Test (500 samples/class)"
echo "========================================"

# Step 0: Verify configuration
echo ""
echo "🔍 Step 0: Verifying configuration..."
python3 verify_config.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration errors detected! Please fix before proceeding."
    exit 1
fi

# Backup current settings
CURRENT_SAMPLES=$(grep "NUM_SAMPLES_PER_CLASS =" config/settings.py | head -1 | awk '{print $3}')
echo "Current samples per class: $CURRENT_SAMPLES"

# Temporarily reduce for quick test
echo ""
echo "⚙️  Temporarily setting NUM_SAMPLES_PER_CLASS = 500 for quick test..."
sed -i.bak "s/NUM_SAMPLES_PER_CLASS = [0-9]*/NUM_SAMPLES_PER_CLASS = 500/" config/settings.py

# Show current configuration
echo ""
echo "📋 Current Configuration:"
grep "COVERT_AMP =" config/settings.py | head -1
grep "RANDOMIZE_SUBCARRIERS =" config/settings.py | head -1
grep "RANDOMIZE_SYMBOLS =" config/settings.py | head -1
grep "ADD_NOISE =" config/settings.py | head -1

# Remove old dataset
echo ""
echo "🗑️  Removing old dataset..."
rm -f dataset/dataset_samples500_sats12.pkl

# Generate new dataset
echo ""
echo "📊 Generating test dataset (500 samples/class)..."
python3 generate_dataset_parallel.py

# Analyze power before training
echo ""
echo "🔍 Analyzing dataset..."
python3 analyze_power.py
python3 check_balance.py

# Train CNN with fewer epochs for quick test
echo ""
echo "🧠 Training CNN (20 epochs for quick feedback)..."
python3 main_detection_cnn.py --epochs 20 --batch-size 32

# Restore original settings
echo ""
echo "♻️  Restoring original settings..."
mv config/settings.py.bak config/settings.py
echo "✓ Restored NUM_SAMPLES_PER_CLASS = $CURRENT_SAMPLES"

echo ""
echo "========================================"
echo "✅ Quick test completed!"
echo "========================================"
echo ""
echo "💡 Next steps based on results:"
echo "   - If AUC < 0.70: Increase COVERT_AMP to 1.4-1.5"
echo "   - If AUC 0.70-0.84: Keep COVERT_AMP = 1.2, train longer"
echo "   - If AUC ≥ 0.85: Reduce COVERT_AMP to 1.0 and add noise"
