"""Data preprocessing and feature engineering."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import (
    CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERICAL_FEATURES,
    RANDOM_SEED
)
from utils import save_model, load_model


class DataPreprocessor:
    """Preprocess features and handle feature engineering."""

    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.binary_features = BINARY_FEATURES
        self.numerical_features = NUMERICAL_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES

    def handle_missing_values(self, X):
        """Handle missing values in dataset."""
        X_clean = X.copy()

        # Fill numerical features with median
        for col in self.numerical_features:
            if col in X_clean.columns and X_clean[col].isnull().sum() > 0:
                X_clean[col].fillna(X_clean[col].median(), inplace=True)

        # Fill categorical features with mode
        for col in self.categorical_features:
            if col in X_clean.columns and X_clean[col].isnull().sum() > 0:
                X_clean[col].fillna(X_clean[col].mode()[0], inplace=True)

        # Fill binary features with 0
        for col in self.binary_features:
            if col in X_clean.columns:
                X_clean[col].fillna(0, inplace=True)

        return X_clean

    def engineer_features(self, X):
        """Create engineered features."""
        X_eng = X.copy()

        # breach_intensity
        if "severity_score" in X_eng.columns and "records_affected_numeric" in X_eng.columns:
            X_eng["breach_intensity"] = (
                X_eng["severity_score"] / np.log(X_eng["records_affected_numeric"] + 1)
            )

        # regulatory_risk_score (weighted combination)
        risk_cols = ["severity_score", "prior_breaches_total", "pii_breach"]
        available_risk_cols = [col for col in risk_cols if col in X_eng.columns]
        if available_risk_cols:
            X_eng["regulatory_risk_score"] = X_eng[available_risk_cols].sum(axis=1)

        # attack_surface (count of attack vectors)
        attack_cols = ["ransomware", "phishing", "malware", "ddos_attack",
                      "insider_threat", "nation_state"]
        available_attack_cols = [col for col in attack_cols if col in X_eng.columns]
        if available_attack_cols:
            X_eng["attack_surface"] = X_eng[available_attack_cols].sum(axis=1)

        return X_eng

    def build_preprocessor(self, X):
        """Build sklearn preprocessing pipeline."""
        # Identify available features
        available_numerical = [f for f in self.numerical_features if f in X.columns]
        available_categorical = [f for f in self.categorical_features if f in X.columns]
        available_binary = [f for f in self.binary_features if f in X.columns]

        # Create transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, available_numerical),
                ("cat", categorical_transformer, available_categorical),
                ("bin", "passthrough", available_binary)
            ],
            remainder="drop"
        )

        return preprocessor, available_numerical, available_categorical, available_binary

    def fit_transform(self, X):
        """Fit preprocessor and transform data."""
        print("Preprocessing data...")

        # Handle missing values
        X_clean = self.handle_missing_values(X)

        # Engineer features
        X_eng = self.engineer_features(X_clean)

        # Build preprocessor
        self.preprocessor, self.num_features, self.cat_features, self.bin_features = \
            self.build_preprocessor(X_eng)

        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X_eng)

        # Get feature names
        self._get_feature_names(X_eng)

        print(f"✓ Data preprocessed: {X_transformed.shape[0]} rows, {X_transformed.shape[1]} features")
        return X_transformed

    def transform(self, X):
        """Transform data using fitted preprocessor."""
        X_clean = self.handle_missing_values(X)
        X_eng = self.engineer_features(X_clean)
        X_transformed = self.preprocessor.transform(X_eng)
        return X_transformed

    def _get_feature_names(self, X):
        """Get names of all features after transformation."""
        feature_names = []

        # Numerical features (scaled)
        feature_names.extend(self.num_features)

        # Categorical features (one-hot encoded)
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        if self.cat_features:
            cat_feature_names = cat_encoder.get_feature_names_out(self.cat_features)
            feature_names.extend(cat_feature_names)

        # Binary features (unchanged)
        feature_names.extend(self.bin_features)

        self.feature_names = feature_names

    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names

    def save(self, filepath):
        """Save preprocessor to file."""
        save_model(self, filepath)
        print(f"✓ Preprocessor saved to {filepath}")

    def load(self, filepath):
        """Load preprocessor from file."""
        loaded = load_model(filepath)
        self.preprocessor = loaded.preprocessor
        self.feature_names = loaded.feature_names
        print(f"✓ Preprocessor loaded from {filepath}")
