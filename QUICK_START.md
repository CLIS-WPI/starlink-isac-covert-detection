# 🚀 Quick Start Guide

## Scenario A (Single-hop)

```bash
# 1. Generate dataset
python3 generate_dataset_parallel.py --scenario sat --total-samples 5000
# Note: Script automatically creates standard name (no copy needed)

# 2. Train model
python3 main_detection_cnn.py --scenario sat --epochs 30 --batch-size 512

# 3. Check results
cat result/scenario_a/detection_results_cnn.json | python3 -m json.tool | grep -A 5 metrics
```

## Scenario B (Dual-hop)

```bash
# 1. Generate dataset
python3 generate_dataset_parallel.py --scenario ground --total-samples 5000
# Note: Script automatically creates standard name (no copy needed)

# 2. Train model
python3 main_detection_cnn.py --scenario ground --epochs 30 --batch-size 512

# 3. Check results
cat result/scenario_b/detection_results_cnn.json | python3 -m json.tool | grep -A 5 metrics
```

## Complete Pipeline (Both)

```bash
./run_production_pipeline.sh
```

