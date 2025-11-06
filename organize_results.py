#!/usr/bin/env python3
"""
Organize existing results into scenario folders.
Run this once to migrate existing results to the new structure.
"""

import os
import shutil
from pathlib import Path

RESULT_DIR = "result"
MODEL_DIR = "model"

def organize_results():
    """Move existing results to scenario_a folder."""
    print("="*70)
    print("📁 Organizing Results into Scenario Folders")
    print("="*70)
    
    # Create scenario folders
    scenario_a_result = Path(RESULT_DIR) / "scenario_a"
    scenario_b_result = Path(RESULT_DIR) / "scenario_b"
    scenario_a_model = Path(MODEL_DIR) / "scenario_a"
    scenario_b_model = Path(MODEL_DIR) / "scenario_b"
    
    for folder in [scenario_a_result, scenario_b_result, scenario_a_model, scenario_b_model]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    # Move existing results (assuming they are from scenario A)
    result_files = [
        "detection_results_cnn_sat.json",
        "detection_results_cnn_csi_sat.json",
        "run_meta_log_sat.csv",
        "run_meta_log_csi_sat.csv",
    ]
    
    moved_count = 0
    for filename in result_files:
        src = Path(RESULT_DIR) / filename
        if src.exists():
            # Remove _sat suffix for new naming
            if filename.endswith("_sat.json"):
                dst_name = filename.replace("_sat.json", ".json")
            elif filename.endswith("_sat.csv"):
                dst_name = filename.replace("_sat.csv", ".csv")
            else:
                dst_name = filename
            
            dst = scenario_a_result / dst_name
            shutil.move(str(src), str(dst))
            print(f"✓ Moved: {src.name} → scenario_a/{dst_name}")
            moved_count += 1
    
    # Move existing models if any
    model_files = [
        "cnn_detector.keras",
        "cnn_detector_csi.keras",
    ]
    
    for filename in model_files:
        src = Path(MODEL_DIR) / filename
        if src.exists():
            dst = scenario_a_model / filename
            shutil.move(str(src), str(dst))
            print(f"✓ Moved: {src.name} → scenario_a/{filename}")
            moved_count += 1
    
    print(f"\n✅ Organization complete! Moved {moved_count} files.")
    print(f"\n📁 New structure:")
    print(f"  {RESULT_DIR}/scenario_a/ - Scenario A results")
    print(f"  {RESULT_DIR}/scenario_b/ - Scenario B results (empty, ready)")
    print(f"  {MODEL_DIR}/scenario_a/ - Scenario A models")
    print(f"  {MODEL_DIR}/scenario_b/ - Scenario B models (empty, ready)")

if __name__ == "__main__":
    organize_results()

