"""
src/data_preprocessing.py
Reusable functions for loading, splitting, scaling, and handling class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw credit card CSV dataset."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")
    return df


def split_features_target(df: pd.DataFrame, target: str = "Class"):
    """Separate features (X) and target (y)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def stratified_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split preserving class distribution."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save_path: str = None):
    """
    Fit StandardScaler on train set and transform both train and test.
    Optionally save the fitted scaler to disk.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    if save_path:
        joblib.dump(scaler, save_path)
        print(f"Scaler saved → {save_path}")

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to training data only.
    Never apply SMOTE to test data.
    """
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — positives: {y_resampled.sum():,}  negatives: {(y_resampled==0).sum():,}")
    return X_resampled, y_resampled


def compute_scale_pos_weight(y_train) -> float:
    """Compute scale_pos_weight for XGBoost (neg_count / pos_count)."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"scale_pos_weight = {spw:.2f}")
    return spw


def save_processed_data(output_dir: str, **datasets):
    """
    Save multiple DataFrames/Series to CSV in output_dir.
    Usage: save_processed_data(path, X_train=X_train, y_train=y_train, ...)
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, data in datasets.items():
        path = os.path.join(output_dir, f"{name}.csv")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_csv(path, index=False)
        else:
            pd.DataFrame(data).to_csv(path, index=False)
        print(f"Saved {name} → {path}")
