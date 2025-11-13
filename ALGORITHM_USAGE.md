# Pseudocode Usage Guide for Overleaf

## 📁 Available Files

1. **`detection_algorithm.tex`** - Detailed version (recommended for paper)
2. **`detection_algorithm_simple.tex`** - Simplified version (for space-constrained papers)
3. **`detection_algorithm_complete.tex`** - Most detailed version (includes all implementation details)

## 🚀 Quick Start

### Step 1: Add Package to Preamble

In your Overleaf document, add to the preamble (before `\begin{document}`):

```latex
\usepackage[ruled,vlined]{algorithm2e}
```

**Options:**
- `ruled`: Horizontal lines above/below algorithm
- `vlined`: Vertical lines on left side
- Other options: `boxed`, `linesnumbered`, `algo2e` (see algorithm2e documentation)

### Step 2: Copy Algorithm

Copy the algorithm environment from one of the `.tex` files into your paper where you want it to appear (typically in Methodology section).

### Step 3: Reference in Text

In your paper text, reference the algorithm:

```latex
We employ 5-fold stratified cross-validation as detailed in 
Algorithm~\ref{alg:detection} to ensure robust performance assessment.
```

## 📊 Which Version to Use?

### Use **`detection_algorithm.tex`** (Recommended):
- ✅ Good balance of detail and brevity
- ✅ Includes all key steps
- ✅ Suitable for most papers
- ✅ ~40 lines

### Use **`detection_algorithm_simple.tex`**:
- ✅ Very concise (~20 lines)
- ✅ Good for space-constrained papers
- ⚠️ Less implementation detail

### Use **`detection_algorithm_complete.tex`**:
- ✅ Most detailed (includes CNN architecture, hyperparameters)
- ✅ Best for comprehensive methodology sections
- ⚠️ Longer (~60 lines)

## 🎨 Customization

### Change Algorithm Title:
```latex
\caption{Your Custom Title Here}
```

### Change Label:
```latex
\label{alg:your_label}
```

### Adjust Formatting:
```latex
% Remove vertical lines:
\usepackage[ruled]{algorithm2e}

% Add line numbers:
\usepackage[ruled,vlined,linesnumbered]{algorithm2e}

% Boxed style:
\usepackage[boxed]{algorithm2e}
```

## 📝 Example Integration

```latex
\section{Methodology}

\subsection{Detection Framework}

We employ a CNN-based detector evaluated using 5-fold stratified 
cross-validation. The complete procedure is outlined in 
Algorithm~\ref{alg:detection}.

% Paste algorithm here
\begin{algorithm}[t]
\caption{CNN-Based Covert Channel Detection...}
...
\end{algorithm}

The algorithm ensures robust evaluation by training on 80\% of 
data and validating on 20\% in each fold, with metrics aggregated 
across all folds to provide mean and standard deviation.
```

## ✅ Verification Checklist

- [ ] Package `algorithm2e` added to preamble
- [ ] Algorithm copied into paper
- [ ] Label matches reference in text
- [ ] Algorithm appears in correct location (typically Methodology)
- [ ] Caption is descriptive
- [ ] Algorithm compiles without errors

## 🔧 Troubleshooting

### Error: "Undefined control sequence"
- **Fix:** Make sure `\usepackage[ruled,vlined]{algorithm2e}` is in preamble

### Algorithm too wide for column
- **Fix:** Use `\resizebox{\columnwidth}{!}{...}` or use simpler version

### Algorithm appears in wrong location
- **Fix:** Use `[t]` (top), `[b]` (bottom), or `[h]` (here) in `\begin{algorithm}[t]`

## 📚 Additional Resources

- [algorithm2e Documentation](https://ctan.org/pkg/algorithm2e)
- [Overleaf Algorithm Guide](https://www.overleaf.com/learn/latex/Algorithms)

---

**Ready to use!** Just copy-paste into your Overleaf document. 🎓

