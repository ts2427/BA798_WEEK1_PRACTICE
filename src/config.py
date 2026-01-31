"""Configuration and constants for ML pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Data file
RAW_DATA_FILE = RAW_DATA_DIR / "FINAL_DISSERTATION_DATASET_ENRICHED.csv"
TARGET_VARIABLE = "has_any_regulatory_action"

# Random seed for reproducibility
RANDOM_SEED = 42

# Train-test split
TEST_SIZE = 0.2
STRATIFY = True

# Feature groups
CATEGORICAL_FEATURES = [
    "organization_type",
    "breach_severity",
    "fcc_category"
]

BINARY_FEATURES = [
    "pii_breach", "health_breach", "financial_breach", "ip_breach",
    "ransomware", "nation_state", "insider_threat", "ddos_attack",
    "phishing", "malware",
    "is_repeat_offender", "is_first_breach",
    "immediate_disclosure", "delayed_disclosure",
    "large_firm", "high_severity_breach"
]

NUMERICAL_FEATURES = [
    "severity_score", "records_affected_numeric", "total_cves",
    "firm_size_log", "roa", "leverage", "sales_q", "assets",
    "prior_breaches_total", "prior_breaches_1yr", "prior_breaches_3yr",
    "disclosure_delay_days", "total_affected",
    "car_5d", "car_30d", "bhar_5d", "bhar_30d",
    "return_volatility_pre", "return_volatility_post",
    "volatility_change"
]

# Feature engineering
ENGINEERED_FEATURES = {
    "breach_intensity": "severity_score / log(records_affected_numeric + 1)",
    "regulatory_risk_score": "weighted combo of severity + prior breaches + pii",
    "attack_surface": "count of attack vector types",
    "disclosure_speed_days": "disclosure_delay_days (binned if needed)"
}

# Model hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced"
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "scale_pos_weight": 1,
    "eval_metric": "logloss"
}

# Evaluation
CV_FOLDS = 5
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
