#!/bin/bash
# 🚀 Quick Start: CNN Detector Testing

echo "========================================================================"
echo "🚀 CNN DETECTOR QUICK START"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Regenerate dataset with COVERT_AMP=0.55 (~5-7% power diff)"
echo "  2. Train CNN detector"
echo "  3. Compare CNN vs RandomForest"
echo ""
echo "Expected time: 5-10 minutes"
echo "========================================================================"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================================================"
echo "Step 1/3: Regenerating dataset with COVERT_AMP=0.55"
echo "========================================================================"
python3 generate_dataset_parallel.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset generation failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Step 2/3: Training CNN detector"
echo "========================================================================"
python3 main_detection_cnn.py

if [ $? -ne 0 ]; then
    echo "❌ CNN training failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Step 3/3: Comparing detectors (RF vs CNN)"
echo "========================================================================"
python3 compare_detectors.py

echo ""
echo "========================================================================"
echo "✅ QUICK START COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 Check results:"
echo "  - CNN results:        result/detection_results_cnn.json"
echo "  - Comparison:         result/detector_comparison.json"
echo "  - Model saved:        model/cnn_detector.keras"
echo ""
echo "🎯 Next steps:"
echo "  - If CNN AUC > 0.85: Edit settings.py (COVERT_AMP=0.45) and retest"
echo "  - If CNN AUC < 0.75: Try CSI fusion with --use-csi flag"
echo "  - Read: CNN_IMPLEMENTATION_GUIDE.md for details"
echo ""
echo "========================================================================"
