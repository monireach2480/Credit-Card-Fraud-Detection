"""
src/train_model.py
Reusable model training functions: Logistic Regression, Random Forest, XGBoost.
"""

import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np


def train_logistic_regression(X_train, y_train, random_state: int = 42, max_iter: int = 1000):
    """Train a Logistic Regression with class_weight='balanced'."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """Train a Random Forest with class_weight='balanced'."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def train_xgboost(X_train, y_train, scale_pos_weight: float = None, random_state: int = 42, **kwargs):
    """
    Train an XGBoost classifier.
    Pass scale_pos_weight=neg/pos for imbalanced datasets.
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    params.update(kwargs)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    print("XGBoost trained.")
    return model


def tune_xgboost(X_train, y_train, scale_pos_weight: float = None,
                 n_iter: int = 30, cv: int = 3, random_state: int = 42):
    """
    RandomizedSearchCV for XGBoost hyperparameter tuning.
    Scores on PR-AUC (best for imbalanced data).
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
    }

    base_params = dict(
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        base_params["scale_pos_weight"] = scale_pos_weight

    xgb = XGBClassifier(**base_params)
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best PR-AUC (CV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def save_model(model, path: str):
    """Save a model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(path: str):
    """Load a model from disk."""
    model = joblib.load(path)
    print(f"Model loaded ← {path}")
    return model
