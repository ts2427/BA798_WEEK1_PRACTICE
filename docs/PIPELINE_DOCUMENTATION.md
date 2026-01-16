# ML Pipeline Documentation

## Overview

This is a modular, scalable ML pipeline for predicting regulatory action on data breaches. The pipeline uses Random Forest and XGBoost models to classify whether a breach will result in regulatory action from FTC, FCC, or State Attorney Generals.

## Target Variable

**`has_any_regulatory_action`** (Binary Classification)
- Predicts whether a breach results in regulatory action
- Class 0: No regulatory action
- Class 1: Regulatory action taken

## Architecture

The pipeline is organized into 7 core modules:

### 1. **config.py**
Central configuration hub with all constants, paths, and parameters.

**Key contents:**
- Project paths (data, outputs, models)
- Target variable definition
- Feature groups (categorical, numerical, binary)
- Model hyperparameters for Random Forest and XGBoost
- Train-test split ratio (80/20)
- Cross-validation folds (5)

**Usage:**
```python
from config import TARGET_VARIABLE, MODELS_DIR, RF_PARAMS
```

### 2. **data_loader.py**
Loads CSV dataset and performs train-test splitting.

**DataLoader class methods:**
- `load_dataset()` - Load CSV and return DataFrame
- `validate_schema()` - Check for required columns
- `get_target_distribution()` - Show class distribution
- `split_data()` - Stratified train/test split (80/20)
- `save_splits()` - Save splits to processed/ folder

**Usage:**
```python
from data_loader import DataLoader

loader = DataLoader()
df = loader.load_dataset()
X_train, X_test, y_train, y_test = loader.split_data()
```

### 3. **preprocessor.py**
Handles missing values, feature engineering, and data scaling.

**DataPreprocessor class methods:**
- `handle_missing_values()` - Fill NaNs (median for numeric, mode for categorical)
- `engineer_features()` - Create derived features (breach_intensity, regulatory_risk_score, attack_surface)
- `fit_transform()` - Fit preprocessor on training data and transform
- `transform()` - Apply fitted preprocessing to new data
- `get_feature_names()` - Return list of feature names after preprocessing
- `save()` / `load()` - Serialize preprocessor

**Feature Engineering:**
- **breach_intensity** = severity_score / log(records_affected + 1)
- **regulatory_risk_score** = Sum of severity + prior breaches + PII breach
- **attack_surface** = Count of attack vector types

**Usage:**
```python
from preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
preprocessor.save("outputs/models/preprocessor.pkl")
```

### 4. **models.py**
Model definitions for Random Forest and XGBoost.

**RandomForestModel class:**
- `train()` - Fit model on training data
- `predict()` - Make binary predictions
- `predict_proba()` - Get prediction probabilities
- `get_feature_importance()` - Return feature importances
- `save()` / `load()` - Serialize model

**XGBoostModel class:**
- Same interface as RandomForestModel

**Usage:**
```python
from models import RandomForestModel, XGBoostModel

rf = RandomForestModel()
rf.train(X_train_processed, y_train)
predictions = rf.predict(X_test_processed)
probabilities = rf.predict_proba(X_test_processed)
rf.save("outputs/models/random_forest_model.pkl")
```

### 5. **trainer.py**
Orchestrates model training and cross-validation.

**ModelTrainer class:**
- `train_model()` - Train single model
- `cross_validate()` - 5-fold cross-validation
- `train_all_models()` - Train both RF and XGBoost

**Usage:**
```python
from trainer import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(X_train_processed, y_train)
```

### 6. **evaluator.py**
Evaluates models, calculates metrics, and creates visualizations.

**ModelEvaluator class:**
- `calculate_metrics()` - Compute accuracy, precision, recall, F1, ROC-AUC
- `plot_confusion_matrix()` - Generate confusion matrix visualization
- `plot_roc_curves()` - Plot ROC curves for all models
- `plot_feature_importance()` - Visualize top 20 features
- `compare_models()` - Create metrics comparison table
- `generate_classification_report()` - Detailed classification metrics
- `generate_report()` - Full evaluation pipeline

**Usage:**
```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator()
comparison_df = evaluator.generate_report(
    models, X_test_processed, y_test, feature_names
)
```

### 7. **utils.py**
Helper functions for file I/O, timestamps, and logging.

**Functions:**
- `setup_output_dirs()` - Create output folder structure
- `get_timestamp()` - Return formatted timestamp
- `save_model()` - Pickle model to file
- `load_model()` - Load pickled model
- `save_metrics()` - Save metrics to JSON
- `save_predictions()` - Export predictions to CSV
- `log_experiment()` - Log experiment metadata

**Usage:**
```python
from utils import setup_output_dirs, save_model, get_timestamp

setup_output_dirs()
save_model(model, "outputs/models/my_model.pkl")
timestamp = get_timestamp()
```

## Running the Pipeline

### Quick Start

```bash
cd BA798_WEEK1_PRACTICE
jupyter notebook notebooks/05_main_pipeline.ipynb
```

Execute all cells in sequence. The notebook handles:
1. Loading and validating data
2. Splitting into train/test sets
3. Preprocessing and feature engineering
4. Training both models
5. Evaluating and comparing results
6. Saving all outputs

### Step-by-Step (Python)

```python
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator

# Load and split data
loader = DataLoader()
df = loader.load_dataset()
X_train, X_test, y_train, y_test = loader.split_data()

# Preprocess
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train_processed, y_train)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.generate_report(
    models, X_test_processed, y_test,
    preprocessor.get_feature_names()
)
```

## Output Structure

```
outputs/
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── preprocessor.pkl
│
├── metrics/
│   ├── model_comparison_YYYYMMDD_HHMMSS.json
│   ├── random_forest_classification_report.txt
│   ├── xgboost_classification_report.txt
│   └── experiment_log.json
│
└── figures/
    ├── confusion_matrices/
    │   ├── random_forest_confusion_matrix.png
    │   └── xgboost_confusion_matrix.png
    │
    ├── feature_importance/
    │   ├── random_forest_feature_importance.png
    │   └── xgboost_feature_importance.png
    │
    ├── roc_curves/
    │   └── roc_comparison.png
    │
    └── target_distribution.png
```

## Configuration Guide

Edit `src/config.py` to customize:

- **Paths:** `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`
- **Target variable:** `TARGET_VARIABLE = "has_any_regulatory_action"`
- **Features:** `CATEGORICAL_FEATURES`, `NUMERICAL_FEATURES`, `BINARY_FEATURES`
- **Model hyperparameters:** `RF_PARAMS`, `XGB_PARAMS`
- **Train-test split:** `TEST_SIZE = 0.2`
- **Validation:** `CV_FOLDS = 5`

## Troubleshooting

### Issue: Import errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Ensure `sys.path` includes the src directory (done in notebook).

### Issue: File not found
```
FileNotFoundError: FINAL_DISSERTATION_DATASET_ENRICHED.csv
```
**Solution:** Verify CSV is in `data/raw/` folder. Update `RAW_DATA_FILE` in config.py if needed.

### Issue: Memory error on large datasets
**Solution:** Reduce `n_estimators` in `RF_PARAMS` and `XGB_PARAMS`, or sample data.

### Issue: Preprocessing fails
**Solution:** Check for unexpected data types. Update feature lists in config.py to match your data.

## Key Metrics Explained

- **Accuracy:** Proportion of correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall (0-1)
- **ROC-AUC:** Area under ROC curve (0.5-1.0, higher is better)

## Extending the Pipeline

### Adding a new model:
1. Create model class in `models.py` with same interface
2. Add instantiation in `trainer.py`'s `train_all_models()`
3. Evaluator will automatically compare

### Modifying preprocessing:
1. Edit `preprocessor.py`'s `engineer_features()` method
2. Update feature lists in `config.py`
3. Re-run pipeline

### Custom evaluation:
1. Add methods to `ModelEvaluator` class
2. Call from notebook or main pipeline

## References

- RandomForest: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- XGBoost: https://xgboost.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/stable/
