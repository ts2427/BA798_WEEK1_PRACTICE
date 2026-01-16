"""Utility functions for ML pipeline."""

import os
import json
from datetime import datetime
from pathlib import Path
import pickle

from config import MODELS_DIR, METRICS_DIR, FIGURES_DIR


def setup_output_dirs():
    """Create all output directories if they don't exist."""
    dirs = [MODELS_DIR, METRICS_DIR, FIGURES_DIR]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for figures
        if dir_path == FIGURES_DIR:
            (dir_path / "confusion_matrices").mkdir(exist_ok=True)
            (dir_path / "feature_importance").mkdir(exist_ok=True)
            (dir_path / "roc_curves").mkdir(exist_ok=True)


def get_timestamp():
    """Get timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model(model, filepath):
    """Save model to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics_dict, filepath):
    """Save metrics to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4)


def load_metrics(filepath):
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_predictions(predictions_df, filepath):
    """Save predictions to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(filepath, index=False)


def log_experiment(experiment_name, params, metrics):
    """Log experiment details to JSON file."""
    log_data = {
        "experiment_name": experiment_name,
        "timestamp": get_timestamp(),
        "parameters": params,
        "metrics": metrics
    }
    filepath = METRICS_DIR / f"{experiment_name}_log.json"
    save_metrics(log_data, filepath)
