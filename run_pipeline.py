"""
Run the complete ML pipeline for data breach regulatory action prediction.
This script executes all pipeline stages and generates outputs.
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import pipeline modules
from config import (
    RANDOM_SEED, TARGET_VARIABLE, MODELS_DIR,
    METRICS_DIR, FIGURES_DIR
)
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from utils import setup_output_dirs, get_timestamp

# Set random seed
np.random.seed(RANDOM_SEED)

print("="*60)
print("ML PIPELINE: Data Breach Regulatory Action Prediction")
print("="*60)

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
print("\n[1/7] SETUP & CONFIGURATION")
print("-" * 60)
setup_output_dirs()
print("[OK] Output directories created")
print(f"[OK] Random seed set: {RANDOM_SEED}")
print(f"[OK] Target variable: {TARGET_VARIABLE}")

# ============================================================================
# 2. DATA LOADING & EDA
# ============================================================================
print("\n[2/7] DATA LOADING & EXPLORATORY DATA ANALYSIS")
print("-" * 60)

loader = DataLoader()
df = loader.load_dataset()

print(f"[OK] Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
print(f"[OK] Data types: {df.dtypes.nunique()} unique types")
print(f"[OK] Missing values: {df.isnull().sum().sum()} total")

# Validate schema
loader.validate_schema()

# Target distribution
target_dist = loader.get_target_distribution()

# Visualize target
plt.figure(figsize=(8, 5))
plt.bar(range(len(target_dist)), list(target_dist.values()))
plt.xticks(range(len(target_dist)), list(target_dist.keys()))
plt.xlabel('Regulatory Action')
plt.ylabel('Count')
plt.title(f'Target Variable Distribution: {TARGET_VARIABLE}')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Target distribution visualized and saved")

# ============================================================================
# 3. DATA SPLITTING
# ============================================================================
print("\n[3/7] DATA SPLITTING")
print("-" * 60)

X_train, X_test, y_train, y_test = loader.split_data()
loader.save_splits()

print(f"[OK] Train set: {len(X_train)} samples")
print(f"[OK] Test set: {len(X_test)} samples")
print("[OK] Train/test splits saved")

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================
print("\n[4/7] DATA PREPROCESSING & FEATURE ENGINEERING")
print("-" * 60)

preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

preprocessor.save(MODELS_DIR / 'preprocessor.pkl')
print("[OK] Preprocessor fitted and saved")
print(f"[OK] Training set shape: {X_train_processed.shape}")
print(f"[OK] Test set shape: {X_test_processed.shape}")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n[5/7] MODEL TRAINING")
print("-" * 60)

trainer = ModelTrainer()
models = trainer.train_all_models(X_train_processed, y_train.values)

for model_name, model in models.items():
    filepath = MODELS_DIR / f'{model_name}_model.pkl'
    model.save(filepath)
    print(f"[OK] {model_name.upper()} model trained and saved")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n[6/7] MODEL EVALUATION & VISUALIZATION")
print("-" * 60)

evaluator = ModelEvaluator()
feature_names = preprocessor.get_feature_names()

comparison_df = evaluator.generate_report(
    models, X_test_processed, y_test.values, feature_names
)

print("\nModel Comparison Results:")
print(comparison_df.round(4).to_string())

# ============================================================================
# 7. SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n[7/7] SUMMARY & RECOMMENDATIONS")
print("-" * 60)

best_model = comparison_df['roc_auc'].idxmax()
best_auc = comparison_df['roc_auc'].max()

print(f"\n{'='*60}")
print("PIPELINE EXECUTION COMPLETE")
print(f"{'='*60}")

print(f"\nRESULTS SUMMARY:")
print(f"   + Data loaded: {len(df):,} records")
print(f"   + Training set: {len(X_train):,} samples")
print(f"   + Test set: {len(X_test):,} samples")
print(f"   + Features after preprocessing: {X_train_processed.shape[1]}")
print(f"   + Models trained: 2 (Random Forest, XGBoost)")

print(f"\nBEST MODEL: {best_model.upper()}")
print(f"   + ROC-AUC: {best_auc:.4f}")
print(f"   + Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
print(f"   + Precision: {comparison_df.loc[best_model, 'precision']:.4f}")
print(f"   + Recall: {comparison_df.loc[best_model, 'recall']:.4f}")
print(f"   + F1-Score: {comparison_df.loc[best_model, 'f1']:.4f}")

print(f"\nOUTPUT FILES SAVED:")
print(f"   + Models: {MODELS_DIR}")
print(f"   + Metrics: {METRICS_DIR}")
print(f"   + Figures: {FIGURES_DIR}")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")
