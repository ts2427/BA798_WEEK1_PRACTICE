"""Data loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR, TARGET_VARIABLE,
    TEST_SIZE, STRATIFY, RANDOM_SEED
)


class DataLoader:
    """Load and split data for ML pipeline."""

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        """Load CSV dataset."""
        print(f"Loading dataset from {RAW_DATA_FILE}...")
        self.df = pd.read_csv(RAW_DATA_FILE)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df

    def validate_schema(self):
        """Validate dataset has required columns."""
        required_cols = [TARGET_VARIABLE]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print("[OK] Schema validation passed")
        return True

    def get_target_distribution(self):
        """Get distribution of target variable."""
        if self.df is None:
            self.load_dataset()

        distribution = self.df[TARGET_VARIABLE].value_counts().to_dict()
        print(f"Target distribution: {distribution}")
        return distribution

    def split_data(self):
        """Split data into train/test sets (stratified)."""
        if self.df is None:
            self.load_dataset()

        X = self.df.drop(columns=[TARGET_VARIABLE])
        y = self.df[TARGET_VARIABLE]

        if STRATIFY:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                stratify=y,
                random_state=RANDOM_SEED
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def save_splits(self):
        """Save train/test splits to CSV."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
        self.X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
        self.y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
        self.y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

        print(f"[OK] Splits saved to {PROCESSED_DATA_DIR}")
