# Publication Figures - Cross-Validation Based

این figures از **Cross-Validation Results** استفاده می‌کنند و آماده برای استفاده در paper هستند.

## 📊 Available Figures

### Figure 1: ROC Curves (CNN from CV + Baselines)
- **Files:** `figure1_roc_curves_cv.pdf`, `figure1_roc_curves_cv.png`
- **Content:** ROC curves comparing CNN (from 5-fold CV) with 3 baseline methods
- **Results:**
  - Scenario A: CNN AUC = 0.62 ± 0.08 (covert attack)
  - Scenario B: CNN AUC = 1.00 ± 0.00 (perfect detection)

### Figure 2: Confusion Matrices (CV Averaged)
- **Files:** `figure2_confusion_matrices_cv.pdf`, `figure2_confusion_matrices_cv.png`
- **Content:** Confusion matrices averaged across 5 CV folds
- **Results:**
  - Scenario A: Accuracy = 67.5% (challenging detection)
  - Scenario B: Accuracy = 100.0% (perfect detection)

### Figure 3: Cross-Validation Box Plots ⭐ (NEW!)
- **Files:** `figure3_cv_boxplots.pdf`, `figure3_cv_boxplots.png`
- **Content:** AUC distribution across 5 folds showing variance
- **Results:**
  - Scenario A: High variance (0.53-0.73), Mean = 0.62 ± 0.08
  - Scenario B: No variance (all 1.00), Mean = 1.00 ± 0.00

## 🎯 How to Use

### For LaTeX Papers:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/figure1_roc_curves_cv.pdf}
  \caption{ROC curves comparing CNN (5-fold CV) with baseline methods.}
  \label{fig:roc_cv}
\end{figure}
```

### For Word/PowerPoint:
- Use PNG files (300 DPI, high quality)
- Already optimized for presentations

## ✅ Quality Assurance

- ✅ All figures use **Cross-Validation Results**
- ✅ CNN metrics: Mean ± Std from 5 folds
- ✅ Baselines: From stored single-split results
- ✅ PDF: Vector format (scalable)
- ✅ PNG: 300 DPI raster format

## 📝 Recommended Captions

### Figure 1:
"ROC curves comparing CNN (evaluated with 5-fold stratified cross-validation) against three baseline methods for both attack scenarios. Scenario A demonstrates challenging detection with AUC = 0.62±0.08, while Scenario B achieves perfect detection with AUC = 1.00±0.00."

### Figure 2:
"Confusion matrices for the CNN detector, averaged across 5 cross-validation folds. Results demonstrate scenario-dependent detectability: Scenario A achieves 67.5% accuracy (validating attack covertness), while Scenario B achieves perfect 100% accuracy."

### Figure 3:
"Box plots showing AUC distribution across 5 cross-validation folds. Scenario A exhibits variance (0.62±0.08) indicating detection challenges, while Scenario B shows perfect consistency (1.00±0.00) across all folds, demonstrating reliable detection capability."

## 🗑️ Removed Files

The following files were based on single-split results (misleading for Scenario A) and have been removed:
- ~~`figure1_roc_curves.pdf/png`~~ (CNN AUC=0.99 was a lucky split!)
- ~~`figure2_confusion_matrices.pdf/png`~~ (Accuracy=94.5% was misleading!)

## 🔄 Regeneration

To regenerate figures:
```bash
python3 generate_cv_figures.py
```

Requirements:
- `/workspace/result/cross_validation_results.json` (CV results)
- `/workspace/result/scenario_*/baseline_results.json` (baseline results)
- `/workspace/dataset/dataset_scenario_*_10000.pkl` (datasets)

---

*Generated: 2025-11-11*  
*Script: generate_cv_figures.py*  
*Status: ✅ Ready for publication*

