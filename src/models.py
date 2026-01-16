"""ML models: Random Forest and XGBoost."""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import RF_PARAMS, XGB_PARAMS
from utils import save_model, load_model


class RandomForestModel:
    """Random Forest classifier wrapper."""

    def __init__(self, **kwargs):
        """Initialize Random Forest with parameters."""
        params = {**RF_PARAMS, **kwargs}
        self.model = RandomForestClassifier(**params)
        self.feature_importance = None

    def train(self, X, y):
        """Train Random Forest model."""
        print("Training Random Forest...")
        self.model.fit(X, y)
        self._get_feature_importance(X)
        print("✓ Random Forest training complete")
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def _get_feature_importance(self, X):
        """Calculate feature importance."""
        self.feature_importance = dict(
            zip(range(X.shape[1]), self.model.feature_importances_)
        )

    def get_feature_importance(self):
        """Return feature importance dictionary."""
        return self.feature_importance

    def save(self, filepath):
        """Save model to file."""
        save_model(self.model, filepath)
        print(f"✓ Random Forest model saved to {filepath}")

    def load(self, filepath):
        """Load model from file."""
        self.model = load_model(filepath)
        print(f"✓ Random Forest model loaded from {filepath}")


class XGBoostModel:
    """XGBoost classifier wrapper."""

    def __init__(self, **kwargs):
        """Initialize XGBoost with parameters."""
        params = {**XGB_PARAMS, **kwargs}
        self.model = XGBClassifier(**params)
        self.feature_importance = None

    def train(self, X, y):
        """Train XGBoost model."""
        print("Training XGBoost...")
        self.model.fit(X, y)
        self._get_feature_importance(X)
        print("✓ XGBoost training complete")
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def _get_feature_importance(self, X):
        """Calculate feature importance."""
        self.feature_importance = dict(
            zip(range(X.shape[1]), self.model.feature_importances_)
        )

    def get_feature_importance(self):
        """Return feature importance dictionary."""
        return self.feature_importance

    def save(self, filepath):
        """Save model to file."""
        save_model(self.model, filepath)
        print(f"✓ XGBoost model saved to {filepath}")

    def load(self, filepath):
        """Load model from file."""
        self.model = load_model(filepath)
        print(f"✓ XGBoost model loaded from {filepath}")
