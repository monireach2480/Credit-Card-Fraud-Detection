"""
src/utils.py
General-purpose utility functions shared across notebooks and modules.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


# ── Model / Artifact I/O ──────────────────────────────────────────────────────

def save_artifact(obj, path: str):
    """Save any Python object to disk with joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved → {path}")


def load_artifact(path: str):
    """Load a joblib artifact from disk."""
    obj = joblib.load(path)
    print(f"Loaded ← {path}")
    return obj


def save_json(obj, path: str):
    """Save a dict/list as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON → {path}")


def load_json(path: str):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


# ── Prediction Pipeline ───────────────────────────────────────────────────────

def predict_transaction(input_data: dict, model, scaler, feature_columns: list,
                         threshold: float) -> dict:
    """
    End-to-end prediction function for a single transaction.

    Args:
        input_data: dict of {feature_name: value}
        model: trained XGBoost (or sklearn) classifier
        scaler: fitted StandardScaler
        feature_columns: ordered list of feature names
        threshold: decision threshold (float)

    Returns:
        dict with prediction, fraud_probability, label
    """
    df = pd.DataFrame([input_data])
    df = df[feature_columns]
    X_scaled = scaler.transform(df)
    fraud_prob = float(model.predict_proba(X_scaled)[0][1])
    prediction = int(fraud_prob >= threshold)
    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Not Fraud",
        "fraud_probability": round(fraud_prob, 6),
        "threshold": threshold,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def save_results_csv(results: list[dict], path: str):
    """Save a list of metric dicts as a CSV table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Results saved → {path}")


def display_class_distribution(y, label: str = "Target"):
    """Print class counts and fraud rate."""
    counts = y.value_counts()
    total = len(y)
    print(f"\n{label} distribution:")
    print(f"  Non-Fraud (0): {counts.get(0, 0):>8,}  ({counts.get(0,0)/total*100:.2f}%)")
    print(f"  Fraud     (1): {counts.get(1, 0):>8,}  ({counts.get(1,0)/total*100:.4f}%)")


def set_plot_style(style: str = "seaborn-v0_8-darkgrid"):
    """Apply a consistent matplotlib style across all notebooks."""
    import matplotlib.pyplot as plt
    plt.style.use(style)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "sans-serif",
    })
