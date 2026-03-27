"""
src/evaluate.py
Reusable model evaluation functions for imbalanced classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, threshold: float = 0.5, model_name: str = "Model") -> dict:
    """
    Full evaluation of a classifier on test data.
    Returns a dict of all key metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Model": model_name,
        "Threshold": threshold,
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12} {v}")
    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate_model dicts."""
    df = pd.DataFrame(results).set_index("Model")
    return df


def plot_confusion_matrix(model, X_test, y_test, threshold: float = 0.5,
                          title: str = "Confusion Matrix", save_path: str = None):
    """Plot a nicely styled confusion matrix."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted Not Fraud", "Predicted Fraud"],
                yticklabels=["Actual Not Fraud", "Actual Fraud"])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for multiple models.
    models_dict: {"ModelName": model_object, ...}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curve(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot Precision-Recall curves for multiple models.
    PR-AUC is more informative than ROC-AUC for imbalanced datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", linewidth=2)

    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def tune_threshold(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    """
    Evaluate model at multiple thresholds. Returns DataFrame with metrics per threshold.
    Useful for choosing the optimal business threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    y_prob = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": round(t, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)
