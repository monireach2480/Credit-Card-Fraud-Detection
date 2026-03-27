"""
export_figures.py — Run this script to extract and save all figures from the trained notebooks.

This script re-runs the key plotting cells from each notebook and saves all figures
to reports/figures/ without you needing to manually edit the .ipynb files.

Usage:
    python export_figures.py

Requirements: Run AFTER notebooks 01–05 have been executed (so models and data exist).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED = os.path.join(DATA_DIR, "processed")
MODELS_FINAL = os.path.join(ROOT, "models", "final")
MODELS_BASELINE = os.path.join(ROOT, "models", "baseline")
DEPLOY = os.path.join(ROOT, "deployment_artifacts")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"figure.figsize": (10, 6), "axes.titlesize": 13})


def fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


print("Loading data and models…")

# ── Load test data ─────────────────────────────────────────────────────────────
X_test = pd.read_csv(os.path.join(PROCESSED, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED, "y_test.csv")).squeeze()

# ── Load models ────────────────────────────────────────────────────────────────
xgb_model = joblib.load(os.path.join(MODELS_FINAL, "best_xgboost_model.pkl"))
lr_model   = joblib.load(os.path.join(MODELS_BASELINE, "logistic_regression_baseline.pkl"))
rf_model   = joblib.load(os.path.join(MODELS_BASELINE, "random_forest_baseline.pkl"))

threshold = joblib.load(os.path.join(DEPLOY, "threshold.pkl"))
feature_columns = joblib.load(os.path.join(DEPLOY, "feature_columns.pkl"))

# align columns
X_test = X_test[feature_columns] if set(feature_columns).issubset(X_test.columns) else X_test

print("✓ All artifacts loaded")

# ── 1. Class Distribution (EDA) ───────────────────────────────────────────────
print("Saving figure: class_distribution.png")
raw_path = os.path.join(DATA_DIR, "raw", "creditcard.csv")
if os.path.exists(raw_path):
    df_raw = pd.read_csv(raw_path, usecols=["Class"])
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df_raw["Class"].value_counts()
    colors = ["#3d5af1", "#e84545"]
    bars = ax.bar(["Not Fraud (0)", "Fraud (1)"], counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,}\n({val/len(df_raw)*100:.3f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_title("Class Distribution — Severe Imbalance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(fig_path("01_class_distribution.png"), dpi=150)
    plt.close()

# ── 2. Confusion Matrix — XGBoost ─────────────────────────────────────────────
print("Saving figure: confusion_matrix_xgboost.png")
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_xgb)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Not Fraud", "Predicted Fraud"],
            yticklabels=["Actual Not Fraud", "Actual Fraud"],
            linewidths=0.5, linecolor="white")
ax.set_title("Confusion Matrix — XGBoost (Final Model)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_path("02_confusion_matrix_xgboost.png"), dpi=150)
plt.close()

# ── 3. ROC Curve — all models ─────────────────────────────────────────────────
print("Saving figure: roc_curve_comparison.png")
fig, ax = plt.subplots(figsize=(8, 6))
for name, model, color in [
    ("XGBoost (Final)", xgb_model, "#3d5af1"),
    ("Logistic Regression", lr_model, "#e84545"),
    ("Random Forest", rf_model, "#00b894"),
]:
    try:
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    except Exception as e:
        print(f"  Skipping {name}: {e}")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_title("ROC Curve — Model Comparison", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path("03_roc_curve_comparison.png"), dpi=150)
plt.close()

# ── 4. Precision-Recall Curve ─────────────────────────────────────────────────
print("Saving figure: pr_curve_comparison.png")
fig, ax = plt.subplots(figsize=(8, 6))
for name, model, color in [
    ("XGBoost (Final)", xgb_model, "#3d5af1"),
    ("Logistic Regression", lr_model, "#e84545"),
    ("Random Forest", rf_model, "#00b894"),
]:
    try:
        prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", color=color, linewidth=2)
    except Exception as e:
        print(f"  Skipping {name}: {e}")

ax.set_title("Precision-Recall Curve — Model Comparison", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path("04_pr_curve_comparison.png"), dpi=150)
plt.close()

# ── 5. Model Comparison Bar Chart ─────────────────────────────────────────────
print("Saving figure: model_comparison_bar.png")
results_path = os.path.join(ROOT, "reports", "tables", "final_model_comparison.csv")
if os.path.exists(results_path):
    df_res = pd.read_csv(results_path)
    metrics = [c for c in df_res.columns if c.lower() not in ["model", "threshold"]]
    if "Model" in df_res.columns and metrics:
        df_plot = df_res.set_index("Model")[metrics]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_plot.index))
        w = 0.8 / len(metrics)
        colors_m = ["#3d5af1", "#e84545", "#00b894", "#fdcb6e", "#a29bfe"]
        for i, metric in enumerate(metrics):
            ax.bar(x + i * w, df_plot[metric], width=w, label=metric,
                   color=colors_m[i % len(colors_m)], edgecolor="white", linewidth=0.8)
        ax.set_xticks(x + w * (len(metrics) - 1) / 2)
        ax.set_xticklabels(df_plot.index, rotation=20, ha="right")
        ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path("05_model_comparison_bar.png"), dpi=150)
        plt.close()

# ── 6. SHAP Global Feature Importance ─────────────────────────────────────────
print("Saving figure: shap_global_importance.png")
shap_path = os.path.join(ROOT, "reports", "tables", "shap_global_feature_importance.csv")
if os.path.exists(shap_path):
    df_shap = pd.read_csv(shap_path).head(20)
    col_feature = df_shap.columns[0]
    col_value = df_shap.columns[1]
    df_shap = df_shap.sort_values(col_value)
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(df_shap[col_feature], df_shap[col_value],
                   color="#3d5af1", edgecolor="white", linewidth=0.5)
    ax.set_title("Global Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path("06_shap_global_importance.png"), dpi=150)
    plt.close()

print(f"\n✅ All figures saved to: {FIGURES_DIR}")
print("Files:")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  └── {f}")
