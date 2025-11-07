#!/bin/bash
# ======================================
# 🚀 Fresh Start: Clean + Run Test
# ======================================
# Single command to clean everything and run from scratch
# ======================================

echo "======================================================================"
echo "🚀 FRESH START: Clean + Run Complete Test"
echo "======================================================================"
echo ""

# Step 1: Clean
echo "🧹 Step 1: Cleaning cache, datasets, and models..."
bash cleanup_all.sh

echo ""
echo "======================================================================"
echo "🚀 Step 2: Running complete test from scratch..."
echo "======================================================================"
echo ""

# Step 2: Run test with logging
bash run_test_with_logging.sh

echo ""
echo "======================================================================"
echo "✅ Fresh Start Complete!"
echo "======================================================================"
echo "📝 Log file location will be shown above"
echo "======================================================================"

