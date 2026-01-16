"""Model evaluation and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

from config import FIGURES_DIR, METRICS_DIR
from utils import save_metrics, get_timestamp


class ModelEvaluator:
    """Evaluate models and create visualizations."""

    def __init__(self):
        self.metrics_summary = {}
        sns.set_style("whitegrid")

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        filepath = FIGURES_DIR / "confusion_matrices" / f"{model_name.lower()}_confusion_matrix.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Confusion matrix saved: {filepath}")

    def plot_roc_curves(self, models_dict, X_test, y_test):
        """Plot and save ROC curves for all models."""
        plt.figure(figsize=(10, 8))

        for model_name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Model Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()

        filepath = FIGURES_DIR / "roc_curves" / "roc_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] ROC curves saved: {filepath}")

    def plot_feature_importance(self, model, model_name, feature_names, top_n=20):
        """Plot and save feature importance."""
        importance = model.get_feature_importance()

        if importance is None:
            print(f"No feature importance for {model_name}")
            return

        # Sort and get top N
        sorted_idx = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                   for idx, _ in sorted_idx]
        values = [val for _, val in sorted_idx]

        plt.figure(figsize=(10, 8))
        plt.barh(features, values)
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importance - {model_name}")
        plt.tight_layout()

        filepath = FIGURES_DIR / "feature_importance" / f"{model_name.lower()}_feature_importance.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Feature importance saved: {filepath}")

    def compare_models(self, models_dict, X_test, y_test):
        """Compare all models and return metrics table."""
        results = {}

        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            results[model_name] = metrics

            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name)

        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        self.metrics_summary = results

        return comparison_df

    def generate_classification_report(self, y_true, y_pred, model_name):
        """Generate and save classification report."""
        report = classification_report(y_true, y_pred, output_dict=False)
        filepath = METRICS_DIR / f"{model_name}_classification_report.txt"

        with open(filepath, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

        print(f"[OK] Classification report saved: {filepath}")

    def save_metrics(self, metrics_dict):
        """Save all metrics to JSON."""
        timestamp = get_timestamp()
        filepath = METRICS_DIR / f"model_comparison_{timestamp}.json"
        save_metrics(metrics_dict, filepath)
        print(f"[OK] Metrics saved: {filepath}")

    def generate_report(self, models_dict, X_test, y_test, feature_names):
        """Generate complete evaluation report."""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60 + "\n")

        # Compare models
        comparison_df = self.compare_models(models_dict, X_test, y_test)
        print("\nModel Comparison:")
        print(comparison_df)

        # Plot ROC curves
        self.plot_roc_curves(models_dict, X_test, y_test)

        # Feature importance for each model
        for model_name, model in models_dict.items():
            self.plot_feature_importance(model, model_name, feature_names)

        # Classification reports
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            self.generate_classification_report(y_test, y_pred, model_name)

        # Save metrics
        self.save_metrics(self.metrics_summary)

        return comparison_df
