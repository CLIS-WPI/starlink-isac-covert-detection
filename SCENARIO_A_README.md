# 🚀 Scenario A: Complete Pipeline

## Quick Start

### Option 1: Bash Script
```bash
./run_scenario_a.sh
```

### Option 2: Python Script
```bash
python3 run_scenario_a.py
```

## What It Does

1. **Generates Dataset** (5000 samples)
   - Single-hop downlink channel
   - Insider@Satellite
   - Output: `dataset/dataset_scenario_a_5000.pkl`

2. **Trains CNN Model** (30 epochs)
   - Batch size: 512
   - Output: `model/scenario_a/cnn_detector.keras`

3. **Generates Results Report**
   - Performance metrics (AUC, Precision, Recall, F1)
   - Power analysis
   - Output: `result/scenario_a/detection_results_cnn.json`

## Expected Output

- **Dataset**: ~79 MB, 5000 samples
- **Model**: `model/scenario_a/cnn_detector.keras`
- **Results**: `result/scenario_a/detection_results_cnn.json`
- **Log**: `training_scenario_a.log`

## Expected Results

- **AUC**: ~0.49-0.67
- **Precision**: ~0.50-0.59
- **Recall**: ~0.95-1.00
- **F1**: ~0.67-0.73
- **Power Diff**: < 0.2% (ultra-covert)

## Manual Steps (if needed)

```bash
# 1. Generate dataset
python3 generate_dataset_parallel.py --scenario sat --total-samples 5000

# 2. Train model
python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512

# 3. Check results
cat result/scenario_a/detection_results_cnn.json | python3 -m json.tool | grep -A 10 metrics
```

