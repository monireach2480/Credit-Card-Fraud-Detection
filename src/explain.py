"""
src/explain.py
Reusable SHAP-based explainability functions for the fraud detection model.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_shap_explainer(model, X_background=None):
    """
    Create a SHAP TreeExplainer for XGBoost/tree models.
    X_background: optional background sample for faster computation.
    """
    if X_background is not None:
        explainer = shap.TreeExplainer(model, X_background)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X):
    """Compute SHAP values for a dataset."""
    shap_values = explainer.shap_values(X)
    return shap_values


def plot_shap_summary(shap_values, X, title: str = "Global Feature Importance (SHAP)",
                      max_display: int = 20, save_path: str = None):
    """
    Plot SHAP summary (beeswarm) — shows global feature importance and direction of impact.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar(shap_values, X, title: str = "Mean |SHAP| Feature Importance",
                  max_display: int = 20, save_path: str = None):
    """
    Plot SHAP bar chart — mean absolute SHAP values per feature.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall(explainer, X, index: int = 0,
                        title: str = "Local Explanation (Waterfall)",
                        save_path: str = None):
    """
    SHAP waterfall plot for a single prediction, showing how each feature
    pushed the model output higher or lower.
    """
    shap_values_obj = explainer(X)
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(shap_values_obj[index], show=False)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_global_importance_df(shap_values, feature_names) -> pd.DataFrame:
    """
    Return a sorted DataFrame of mean |SHAP| values per feature.
    Useful for saving to reports/tables/.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df
