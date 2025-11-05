#!/bin/bash

# 🚀 Advanced CNN Optimization - Full Pipeline
# این اسکریپت تمام مراحل optimization رو اجرا می‌کنه

set -e  # Exit on error

# رنگ‌ها برای output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Advanced CNN Optimization Pipeline${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Step 1: Verify Configuration
echo -e "${YELLOW}📋 Step 1: Verifying Configuration...${NC}"
python3 verify_config.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Configuration verification failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Configuration verified${NC}"
echo ""

# Step 2: Quick Diagnostics on Existing Dataset
echo -e "${YELLOW}🔍 Step 2: Running Quick Diagnostics...${NC}"
if [ -f "data/covert_dataset_train.pkl" ]; then
    echo "Found existing dataset, running diagnostics..."
    python3 diagnose_advanced.py
    echo -e "${GREEN}✅ Diagnostics complete${NC}"
else
    echo -e "${YELLOW}⚠️  No existing dataset found, skipping diagnostics${NC}"
fi
echo ""

# Prompt user for test type
echo -e "${BLUE}Choose test mode:${NC}"
echo "  1. Quick Test (baseline + CSI + full, ~1 hour)"
echo "  2. Full Ablation Study (all 7 configs × 3 runs, ~8 hours)"
echo "  3. Single Full Training (3000 samples, all features, ~45 min)"
echo "  4. Skip tests, just generate dataset"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        # Quick Ablation Study
        echo -e "${YELLOW}🧪 Step 3: Quick Ablation Study...${NC}"
        echo "Testing: baseline, +CSI, full config"
        python3 ablation_study.py --quick
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Quick ablation study complete${NC}"
            echo "Check 'ablation_results.txt' for detailed results"
        else
            echo -e "${RED}❌ Ablation study failed${NC}"
            exit 1
        fi
        ;;
    
    2)
        # Full Ablation Study
        echo -e "${YELLOW}🧪 Step 3: Full Ablation Study...${NC}"
        echo "Testing all 7 configs with 3 runs each"
        echo "This will take ~8 hours..."
        read -p "Continue? (y/n): " confirm
        
        if [ "$confirm" = "y" ]; then
            python3 ablation_study.py --runs 3
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Full ablation study complete${NC}"
                echo "Check 'ablation_results.txt' for detailed results"
            else
                echo -e "${RED}❌ Ablation study failed${NC}"
                exit 1
            fi
        else
            echo "Skipping full ablation study"
        fi
        ;;
    
    3)
        # Single Full Training
        echo -e "${YELLOW}🎯 Step 3: Full Training with All Features...${NC}"
        
        # Generate dataset with 3000 samples
        echo "Generating dataset with 3000 samples per class..."
        python3 generate_dataset_parallel.py
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Dataset generation failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Dataset generated${NC}"
        
        # Train with all features
        echo "Training CNN with all advanced features..."
        python3 main_detection_cnn.py --epochs 50
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Training complete${NC}"
        else
            echo -e "${RED}❌ Training failed${NC}"
            exit 1
        fi
        ;;
    
    4)
        # Just generate dataset
        echo -e "${YELLOW}📊 Step 3: Generating Dataset Only...${NC}"
        python3 generate_dataset_parallel.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Dataset generated${NC}"
        else
            echo -e "${RED}❌ Dataset generation failed${NC}"
            exit 1
        fi
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}🎉 Pipeline Complete!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Show summary
echo -e "${YELLOW}📊 Summary:${NC}"
echo "  - Configuration: Verified ✅"
echo "  - Dataset: Generated ✅"

if [ $choice -eq 1 ] || [ $choice -eq 2 ]; then
    echo "  - Ablation Study: Complete ✅"
    echo ""
    echo "📁 Check results:"
    echo "  - ablation_results.txt"
    echo "  - ablation_summary.json"
fi

if [ $choice -eq 3 ]; then
    echo "  - Training: Complete ✅"
    echo ""
    echo "📁 Check results:"
    echo "  - model/cnn_detector.keras"
    echo "  - logs/training_history.png"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
if [ $choice -eq 1 ]; then
    echo "  - Review ablation_results.txt"
    echo "  - Run full ablation study: ./run_advanced_optimization.sh → choice 2"
    echo "  - Or train best config: python3 main_detection_cnn.py --epochs 50"
elif [ $choice -eq 2 ]; then
    echo "  - Review ablation_results.txt for best configuration"
    echo "  - Update config/settings.py with best settings"
    echo "  - Train final model: python3 main_detection_cnn.py --epochs 50"
elif [ $choice -eq 3 ]; then
    echo "  - Evaluate model: python3 diagnose_advanced.py"
    echo "  - Compare with baseline: python3 compare_detectors.py"
elif [ $choice -eq 4 ]; then
    echo "  - Train model: python3 main_detection_cnn.py"
    echo "  - Or run ablation study: python3 ablation_study.py"
fi

echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "  - ADVANCED_OPTIMIZATION_GUIDE.md"
echo "  - CNN_IMPLEMENTATION_GUIDE.md"
echo "  - FINAL_OPTIMAL_SETTINGS.md"
echo ""
