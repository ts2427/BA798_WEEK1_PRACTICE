"""Model training orchestration."""

from sklearn.model_selection import cross_validate

from config import CV_FOLDS, METRICS
from models import RandomForestModel, XGBoostModel


class ModelTrainer:
    """Train and evaluate models."""

    def __init__(self):
        self.trained_models = {}
        self.cv_results = {}

    def train_model(self, model, X_train, y_train):
        """Train a single model."""
        model.train(X_train, y_train)
        return model

    def cross_validate(self, model, X, y, cv=CV_FOLDS):
        """Perform cross-validation."""
        print(f"Running {cv}-fold cross-validation...")
        cv_results = cross_validate(
            model.model,
            X, y,
            cv=cv,
            scoring=METRICS,
            return_train_score=True
        )
        print(f"[OK] Cross-validation complete")
        return cv_results

    def train_all_models(self, X_train, y_train):
        """Train Random Forest and XGBoost models."""
        models = {}

        # Random Forest
        rf_model = RandomForestModel()
        self.train_model(rf_model, X_train, y_train)
        models["random_forest"] = rf_model

        # XGBoost
        xgb_model = XGBoostModel()
        self.train_model(xgb_model, X_train, y_train)
        models["xgboost"] = xgb_model

        self.trained_models = models
        print(f"\n[OK] All models trained")

        return models
