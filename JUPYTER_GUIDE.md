# Jupyter Notebook Guide

**Notebook:** `notebooks/05_main_pipeline.ipynb`
**Purpose:** Interactive ML pipeline execution with visualizations
**Runtime:** ~2-3 minutes on typical hardware

---

## Quick Start

### Option 1: Start Jupyter Server

```bash
# Navigate to project directory
cd BA798_WEEK1_PRACTICE

# Start Jupyter
jupyter notebook

# Open the notebook
notebooks/05_main_pipeline.ipynb
```

### Option 2: Use Jupyter Lab (Modern Interface)

```bash
jupyter lab notebooks/05_main_pipeline.ipynb
```

### Option 3: VS Code (Integrated Jupyter)

```bash
code notebooks/05_main_pipeline.ipynb
```

---

## What the Notebook Does

The notebook is organized into **7 sections** that execute the complete ML pipeline:

### [1/7] Setup & Configuration
- Import all required libraries
- Set random seed for reproducibility
- Configure output directories
- Display target variable definition

### [2/7] Data Loading & EDA
- Load CSV dataset (858 records, 98 features)
- Display dataset info and missing values
- Show target variable distribution
- Create target distribution visualization

### [3/7] Data Splitting
- Perform stratified train/test split (80/20)
- Save splits to CSV files
- Display class distribution in each set

### [4/7] Data Preprocessing
- Handle missing values (median for numerical, mode for categorical)
- Engineer features (breach_intensity, regulatory_risk_score, attack_surface)
- Scale and encode features
- Save preprocessor for reuse

### [5/7] Model Training
- Train Random Forest classifier
- Train XGBoost classifier
- Save both models
- Display training completion

### [6/7] Model Evaluation
- Calculate metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generate confusion matrices (PNG)
- Plot ROC curves comparison
- Plot feature importance (top 20 features per model)
- Save classification reports
- Create metrics comparison table

### [7/7] Summary & Recommendations
- Display best model (XGBoost)
- Show final performance metrics
- List all saved outputs
- Print execution timestamp

---

## Running the Notebook

### Step-by-Step Execution

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```
   Browser opens at `http://localhost:8888`

2. **Open the Notebook**
   Navigate to: `notebooks/05_main_pipeline.ipynb`

3. **Run All Cells**
   - Menu: `Cell → Run All`
   - Keyboard: `Ctrl + Shift + Enter`
   - Or run cells individually with `Ctrl + Enter`

4. **View Outputs**
   - Metrics displayed below each cell
   - Figures displayed inline
   - Results saved to `outputs/` directory

---

## Expected Output

### Terminal Output
```
============================================================
ML PIPELINE: Data Breach Regulatory Action Prediction
============================================================

[1/7] SETUP & CONFIGURATION
[OK] Output directories created
[OK] Random seed set: 42

[2/7] DATA LOADING & EXPLORATORY DATA ANALYSIS
[OK] Dataset loaded: 858 records, 98 features
...
[OK] Target distribution visualized and saved

[3/7] DATA SPLITTING
[OK] Train set: 686 samples
[OK] Test set: 172 samples

[4/7] DATA PREPROCESSING
[OK] Data preprocessed: 686 rows, 49 features

[5/7] MODEL TRAINING
[OK] Random Forest training complete
[OK] XGBoost training complete

[6/7] MODEL EVALUATION
[OK] Confusion matrix saved
[OK] ROC curves saved
[OK] Feature importance saved
[OK] Metrics saved

[7/7] SUMMARY
BEST MODEL: XGBOOST
   ROC-AUC: 0.9961
   Accuracy: 0.9884
   Precision: 0.9231
   Recall: 0.9231
   F1-Score: 0.9231
```

### Jupyter Visualizations

**Inline Plots:**
- Target distribution histogram
- Confusion matrices (one per model)
- ROC curves comparison
- Feature importance bar charts

### Generated Files

**Models (outputs/models/):**
- `random_forest_model.pkl` (546 KB)
- `xgboost_model.pkl` (130 KB)
- `preprocessor.pkl` (6.1 KB)

**Metrics (outputs/metrics/):**
- `model_comparison_*.json` - All metrics in JSON format
- `random_forest_classification_report.txt`
- `xgboost_classification_report.txt`

**Figures (outputs/figures/):**
- `confusion_matrices/random_forest_confusion_matrix.png`
- `confusion_matrices/xgboost_confusion_matrix.png`
- `roc_curves/roc_comparison.png`
- `feature_importance/random_forest_feature_importance.png`
- `feature_importance/xgboost_feature_importance.png`
- `target_distribution.png`

---

## Tips & Tricks

### Running Specific Sections

To run only certain sections:

1. **Setup Only**
   - Run cells from [1/7] only

2. **Data Loading Only**
   - Run cells from [1/7] and [2/7]

3. **Training & Eval**
   - Run all cells up to [5/7], then skip to [6/7] for evaluation

### Debugging

If you encounter errors:

```python
# Check dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

# Check splits
print(X_train.shape, X_test.shape)

# Check preprocessor
print(X_train_processed.shape)
print(feature_names[:10])

# Check models
print(models.keys())
for name, model in models.items():
    print(f"{name}: trained = {model.model is not None}")
```

### Stopping Execution

If you need to stop:
- Press `Ctrl + C` in terminal
- Or click: `Kernel → Interrupt` in Jupyter menu

### Restarting Kernel

To clear all variables and start fresh:
- Menu: `Kernel → Restart`
- Then re-run from [1/7]

---

## System Requirements

**Minimum:**
- Python 3.9+
- 2 GB RAM
- ~5 minutes execution time

**Recommended:**
- Python 3.10+
- 4+ GB RAM
- ~2 minutes execution time

### Installation Check

```python
# In notebook cell, verify all imports work
import pandas as pd
import numpy as np
import scikit-learn
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns

print("All imports successful!")
```

---

## Customization

### Change Random Seed
In cell [1/7], modify:
```python
RANDOM_SEED = 42  # Change this value
```

### Change Train/Test Split
In `src/config.py`, modify:
```python
TEST_SIZE = 0.2  # Change from 80/20 to desired ratio
```

### Add More Features
In `src/preprocessor.py`, add to `engineer_features()`:
```python
X_eng["new_feature"] = X_eng["col1"] / X_eng["col2"]
```

### Change Model Hyperparameters
In `src/config.py`, modify:
```python
RF_PARAMS = {
    "n_estimators": 100,  # Change number of trees
    "max_depth": 15,      # Change tree depth
}
```

---

## Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
# or
uv sync
```

### Issue: "CSV file not found"
**Solution:**
Ensure `data/raw/FINAL_DISSERTATION_DATASET_ENRICHED.csv` exists
```bash
ls -la data/raw/
```

### Issue: "Out of memory"
**Solution:**
- Close other applications
- Reduce dataset size (sample rows)
- Run on machine with more RAM

### Issue: "Models training very slowly"
**Solution:**
Reduce `n_estimators` in `src/config.py`:
```python
RF_PARAMS = {"n_estimators": 50}  # Fewer trees
XGB_PARAMS = {"n_estimators": 50}
```

### Issue: "Can't create output files"
**Solution:**
Ensure write permissions:
```bash
chmod -R 755 outputs/
chmod -R 755 data/
```

---

## Running in Google Colab

To run in Google Colab (cloud-based Jupyter):

```python
# 1. Clone repository
!git clone https://github.com/ts2427/BA798_WEEK1_PRACTICE.git
!cd BA798_WEEK1_PRACTICE && git checkout MLMODEL

# 2. Install dependencies
!pip install -r BA798_WEEK1_PRACTICE/requirements.txt

# 3. Navigate and run
import os
os.chdir('BA798_WEEK1_PRACTICE')

# 4. Execute pipeline script
exec(open('run_pipeline.py').read())
```

---

## Next Steps After Execution

1. **Review Results**
   - Check `RESULTS_ANALYSIS.md` for detailed findings
   - Review `outputs/metrics/` for performance metrics
   - Examine visualizations in `outputs/figures/`

2. **Modify & Rerun**
   - Change hyperparameters in `src/config.py`
   - Run notebook again to see new results
   - Compare metrics between runs

3. **Share Results**
   - Export metrics to CSV
   - Save notebook as HTML (File → Export As)
   - Share `RESULTS_ANALYSIS.md` with stakeholders

4. **Deploy**
   - Use trained models from `outputs/models/`
   - Load preprocessor to transform new data
   - Make predictions on new breach incidents

---

## Keyboard Shortcuts

**Jupyter Shortcuts:**
- `Shift + Enter` - Run cell and move to next
- `Ctrl + Enter` - Run cell in place
- `Alt + Enter` - Run cell and insert new below
- `A` - Insert cell above
- `B` - Insert cell below
- `D, D` - Delete cell
- `M` - Convert to Markdown
- `Y` - Convert to Code

---

## Support

For issues or questions:
1. Check `PIPELINE_DOCUMENTATION.md` for technical details
2. Review `GITIGNORE_GUIDE.md` for file structure
3. Read error messages carefully
4. Check GitHub issues: https://github.com/ts2427/BA798_WEEK1_PRACTICE/issues

---

**Last Updated:** 2026-01-16
**Status:** Ready for interactive execution
