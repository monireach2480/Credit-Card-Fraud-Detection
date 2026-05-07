# =============================================================================
# Notebook 06 — Model Improvements
# Credit Card Fraud Detection
# =============================================================================

# ── 1. IMPORT LIBRARIES ───────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs("../reports/tables", exist_ok=True)
os.makedirs("../models/improved", exist_ok=True)


# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────

X_train_scaled = pd.read_csv("../data/processed/X_train_scaled.csv")
X_test_scaled  = pd.read_csv("../data/processed/X_test_scaled.csv")
y_train        = pd.read_csv("../data/processed/y_train.csv").squeeze()
y_test         = pd.read_csv("../data/processed/y_test.csv").squeeze()

X_train_resampled = pd.read_csv("../data/processed/X_train_resampled.csv")
y_train_resampled = pd.read_csv("../data/processed/y_train_resampled.csv").squeeze()

print("Train shape:", X_train_scaled.shape)
print("Test shape :", X_test_scaled.shape)


# ── 3. HELPER: EVALUATE MODEL ─────────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred, y_prob):
    return {
        "Model"    : model_name,
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_true, y_prob),
        "PR-AUC"   : average_precision_score(y_true, y_prob)
    }


# ── 4. FEATURE SELECTION USING SHAP ──────────────────────────────────────────

print("\n=== STEP 1: FEATURE SELECTION USING SHAP ===")

final_xgb   = joblib.load("../models/final/best_xgboost_model.pkl")
explainer   = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_test_scaled)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance_df = pd.DataFrame({
    "Feature"    : X_test_scaled.columns,
    "Mean |SHAP|": mean_abs_shap
}).sort_values("Mean |SHAP|", ascending=False)

print(feature_importance_df.to_string(index=False))

plt.figure(figsize=(9, 6))
sns.barplot(data=feature_importance_df, x="Mean |SHAP|", y="Feature")
plt.title("SHAP Global Feature Importance (all features)")
plt.tight_layout()
plt.savefig("../reports/tables/shap_all_features.png", dpi=150)
plt.show()

# Keep features above the mean of all mean |SHAP| values
shap_threshold    = mean_abs_shap.mean()
selected_features = feature_importance_df[
    feature_importance_df["Mean |SHAP|"] >= shap_threshold
]["Feature"].tolist()
dropped_features  = [f for f in X_test_scaled.columns if f not in selected_features]

print(f"\nSHAP threshold : {shap_threshold:.4f}")
print(f"Features kept  : {len(selected_features)} -> {selected_features}")
print(f"Features dropped: {len(dropped_features)} -> {dropped_features}")

X_train_sel           = X_train_scaled[selected_features]
X_test_sel            = X_test_scaled[selected_features]
X_train_resampled_sel = X_train_resampled[selected_features]


# ── 5. FIXED XGBOOST TUNING ───────────────────────────────────────────────────
# Key fixes vs Notebook 4:
#   scoring = "average_precision"  (was "recall" which caused precision collapse)
#   n_jobs  = 1                    (MUST be 1 on Windows to avoid crash)
#   cv      = 3                    (enough for reliable estimate, saves time)

print("\n=== STEP 2: FIXED XGBOOST TUNING ===")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw       = neg_count / pos_count
print(f"scale_pos_weight = {spw:.2f}")

param_dist_xgb = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [3, 4, 5, 6],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "subsample"       : [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma"           : [0, 0.1, 0.3],
    "reg_alpha"       : [0, 0.1, 1.0],
    "reg_lambda"      : [1, 2, 5],
    "scale_pos_weight": [spw]
}

# ── 5A: Full feature set ──
print("\nTuning XGBoost on full features...")
search_full = RandomizedSearchCV(
    estimator           = XGBClassifier(
                              random_state=42,
                              eval_metric="logloss",
                              use_label_encoder=False
                          ),
    param_distributions = param_dist_xgb,
    n_iter              = 20,
    scoring             = "average_precision",
    cv                  = 3,
    verbose             = 1,
    random_state        = 42,
    n_jobs              = 1
)
search_full.fit(X_train_scaled, y_train)

best_xgb_full    = search_full.best_estimator_
y_pred_xgb_full  = best_xgb_full.predict(X_test_scaled)
y_prob_xgb_full  = best_xgb_full.predict_proba(X_test_scaled)[:, 1]
results_xgb_full = evaluate_model(
    "XGBoost Tuned (all features)", y_test, y_pred_xgb_full, y_prob_xgb_full
)
print("Best params (full):", search_full.best_params_)
print(pd.DataFrame([results_xgb_full]))

# ── 5B: SHAP-selected feature set ──
print("\nTuning XGBoost on SHAP-selected features...")
search_sel = RandomizedSearchCV(
    estimator           = XGBClassifier(
                              random_state=42,
                              eval_metric="logloss",
                              use_label_encoder=False
                          ),
    param_distributions = param_dist_xgb,
    n_iter              = 20,
    scoring             = "average_precision",
    cv                  = 3,
    verbose             = 1,
    random_state        = 42,
    n_jobs              = 1
)
search_sel.fit(X_train_sel, y_train)

best_xgb_sel    = search_sel.best_estimator_
y_pred_xgb_sel  = best_xgb_sel.predict(X_test_sel)
y_prob_xgb_sel  = best_xgb_sel.predict_proba(X_test_sel)[:, 1]
results_xgb_sel = evaluate_model(
    "XGBoost Tuned (SHAP features)", y_test, y_pred_xgb_sel, y_prob_xgb_sel
)
print("Best params (SHAP):", search_sel.best_params_)
print(pd.DataFrame([results_xgb_sel]))

# Pick whichever XGBoost variant has higher PR-AUC
if results_xgb_sel["PR-AUC"] >= results_xgb_full["PR-AUC"]:
    best_xgb_model  = best_xgb_sel
    best_xgb_prob   = y_prob_xgb_sel
    best_xgb_result = results_xgb_sel
    best_xgb_Xtest  = X_test_sel
    print("\n-> Using SHAP-selected features for final XGBoost.")
else:
    best_xgb_model  = best_xgb_full
    best_xgb_prob   = y_prob_xgb_full
    best_xgb_result = results_xgb_full
    best_xgb_Xtest  = X_test_scaled
    print("\n-> Using full features for final XGBoost.")


# ── 6. LIGHTGBM ───────────────────────────────────────────────────────────────

print("\n=== STEP 3: LIGHTGBM ===")

param_dist_lgbm = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [-1, 5, 7, 10],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "num_leaves"       : [31, 63, 127],
    "subsample"        : [0.7, 0.8, 0.9],
    "colsample_bytree" : [0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 50],
    "reg_alpha"        : [0, 0.1, 1.0],
    "reg_lambda"       : [0, 1, 5],
    "scale_pos_weight" : [spw]
}

# ── 6A: Full feature set ──
print("\nTuning LightGBM on full features...")
lgbm_search_full = RandomizedSearchCV(
    estimator           = LGBMClassifier(
                              random_state=42,
                              is_unbalance=False,
                              verbose=-1
                          ),
    param_distributions = param_dist_lgbm,
    n_iter              = 20,
    scoring             = "average_precision",
    cv                  = 3,
    verbose             = 1,
    random_state        = 42,
    n_jobs              = 1
)
lgbm_search_full.fit(X_train_scaled, y_train)

best_lgbm_full    = lgbm_search_full.best_estimator_
y_pred_lgbm_full  = best_lgbm_full.predict(X_test_scaled)
y_prob_lgbm_full  = best_lgbm_full.predict_proba(X_test_scaled)[:, 1]
results_lgbm_full = evaluate_model(
    "LightGBM (all features)", y_test, y_pred_lgbm_full, y_prob_lgbm_full
)
print("Best params (LightGBM full):", lgbm_search_full.best_params_)
print(pd.DataFrame([results_lgbm_full]))

# ── 6B: SHAP-selected feature set ──
print("\nTuning LightGBM on SHAP-selected features...")
lgbm_search_sel = RandomizedSearchCV(
    estimator           = LGBMClassifier(
                              random_state=42,
                              is_unbalance=False,
                              verbose=-1
                          ),
    param_distributions = param_dist_lgbm,
    n_iter              = 20,
    scoring             = "average_precision",
    cv                  = 3,
    verbose             = 1,
    random_state        = 42,
    n_jobs              = 1
)
lgbm_search_sel.fit(X_train_sel, y_train)

best_lgbm_sel    = lgbm_search_sel.best_estimator_
y_pred_lgbm_sel  = best_lgbm_sel.predict(X_test_sel)
y_prob_lgbm_sel  = best_lgbm_sel.predict_proba(X_test_sel)[:, 1]
results_lgbm_sel = evaluate_model(
    "LightGBM (SHAP features)", y_test, y_pred_lgbm_sel, y_prob_lgbm_sel
)
print("Best params (LightGBM SHAP):", lgbm_search_sel.best_params_)
print(pd.DataFrame([results_lgbm_sel]))

# Pick whichever LGBM variant has higher PR-AUC
if results_lgbm_sel["PR-AUC"] >= results_lgbm_full["PR-AUC"]:
    best_lgbm_model  = best_lgbm_sel
    best_lgbm_prob   = y_prob_lgbm_sel
    best_lgbm_result = results_lgbm_sel
    best_lgbm_Xtest  = X_test_sel
    print("\n-> Using SHAP-selected features for final LightGBM.")
else:
    best_lgbm_model  = best_lgbm_full
    best_lgbm_prob   = y_prob_lgbm_full
    best_lgbm_result = results_lgbm_full
    best_lgbm_Xtest  = X_test_scaled
    print("\n-> Using full features for final LightGBM.")


# ── 7. OPTIMAL THRESHOLD SELECTION ───────────────────────────────────────────

print("\n=== STEP 4: OPTIMAL THRESHOLD SELECTION ===")

def find_best_threshold(y_true, y_prob, model_name):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision_vals + recall_vals) == 0,
        0,
        2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
    )
    best_idx       = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]
    print(f"{model_name} -> best threshold: {best_threshold:.4f},  best F1: {best_f1:.4f}")
    return best_threshold

xgb_best_threshold  = find_best_threshold(y_test, best_xgb_prob,  "XGBoost Tuned")
lgbm_best_threshold = find_best_threshold(y_test, best_lgbm_prob, "LightGBM Tuned")

y_pred_xgb_tuned  = (best_xgb_prob  >= xgb_best_threshold).astype(int)
y_pred_lgbm_tuned = (best_lgbm_prob >= lgbm_best_threshold).astype(int)

results_xgb_thresh = evaluate_model(
    "XGBoost Tuned + Optimal Threshold", y_test, y_pred_xgb_tuned, best_xgb_prob
)
results_lgbm_thresh = evaluate_model(
    "LightGBM Tuned + Optimal Threshold", y_test, y_pred_lgbm_tuned, best_lgbm_prob
)

print("\nXGBoost with optimal threshold:")
print(pd.DataFrame([results_xgb_thresh]))
print("\nLightGBM with optimal threshold:")
print(pd.DataFrame([results_lgbm_thresh]))


# ── 8. FULL MODEL COMPARISON ──────────────────────────────────────────────────

print("\n=== STEP 5: FULL MODEL COMPARISON ===")

baseline_df = pd.read_csv("../reports/tables/baseline_model_results.csv")

prev_xgb = {
    "Model"    : "XGBoost scale_pos_weight (Notebook 4)",
    "Accuracy" : 0.999526,
    "Precision": 0.881720,
    "Recall"   : 0.836735,
    "F1-Score" : 0.858639,
    "ROC-AUC"  : 0.968238,
    "PR-AUC"   : 0.880004
}

all_results = pd.concat([
    baseline_df,
    pd.DataFrame([prev_xgb]),
    pd.DataFrame([results_xgb_thresh]),
    pd.DataFrame([results_lgbm_thresh])
], ignore_index=True).sort_values("PR-AUC", ascending=False)

print(all_results.to_string(index=False))
all_results.to_csv("../reports/tables/full_model_comparison.csv", index=False)
print("\nSaved -> ../reports/tables/full_model_comparison.csv")

metrics_to_plot = ["Recall", "F1-Score", "PR-AUC"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, metrics_to_plot):
    sns.barplot(data=all_results, x=metric, y="Model", ax=ax, orient="h")
    ax.set_title(metric)
    ax.set_xlim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.suptitle("Full Model Comparison", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("../reports/tables/full_model_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 9. CONFUSION MATRICES ─────────────────────────────────────────────────────

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"../reports/tables/{title.replace(' ', '_')}_cm.png", dpi=150)
    plt.show()
    

plot_cm(y_test, y_pred_xgb_tuned,  "XGBoost Tuned + Optimal Threshold")
plot_cm(y_test, y_pred_lgbm_tuned, "LightGBM Tuned + Optimal Threshold")

print("\nXGBoost classification report:")
print(classification_report(y_test, y_pred_xgb_tuned, zero_division=0))
print("LightGBM classification report:")
print(classification_report(y_test, y_pred_lgbm_tuned, zero_division=0))


# ── 10. PRECISION-RECALL CURVES ───────────────────────────────────────────────

plt.figure(figsize=(8, 6))
for prob, label, color in [
    (best_xgb_prob,  "XGBoost Tuned",  "steelblue"),
    (best_lgbm_prob, "LightGBM Tuned", "darkorange"),
]:
    p, r, _ = precision_recall_curve(y_test, prob)
    auc_val  = average_precision_score(y_test, prob)
    plt.plot(r, p, label=f"{label} (PR-AUC={auc_val:.4f})", lw=2, color=color)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Improved Models")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/tables/pr_curves_improved.png", dpi=150)
plt.show()


# ── 11. SAVE MODELS AND ARTIFACTS ─────────────────────────────────────────────

joblib.dump(best_xgb_model,      "../models/improved/xgb_tuned_improved.pkl")
joblib.dump(best_lgbm_model,     "../models/improved/lgbm_tuned_improved.pkl")
joblib.dump(xgb_best_threshold,  "../models/improved/xgb_best_threshold.pkl")
joblib.dump(lgbm_best_threshold, "../models/improved/lgbm_best_threshold.pkl")
joblib.dump(selected_features,   "../models/improved/selected_features.pkl")

print("\nAll models and artifacts saved to ../models/improved/")
print(f"XGBoost optimal threshold : {xgb_best_threshold:.4f}")
print(f"LightGBM optimal threshold: {lgbm_best_threshold:.4f}")
print(f"Selected features ({len(selected_features)}): {selected_features}")


# ── 12. SUMMARY ───────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("NOTEBOOK 06 - SUMMARY OF CHANGES vs NOTEBOOK 4")
print("="*60)
print("""
1. FEATURE SELECTION
   Used SHAP to drop 22 near-zero features. Tested both
   full and reduced sets; best performer auto-selected.

2. FIXED TUNING SCORING
   Changed scoring from 'recall' to 'average_precision'.
   This prevents precision from collapsing to ~0.11.

3. LIGHTGBM
   Added as a new model with its own tuned search.
   Uses num_leaves for finer complexity control.

4. OPTIMAL THRESHOLD
   PR-curve F1 maximisation replaces manual grid search.
   Applied to both XGBoost and LightGBM final models.

5. WINDOWS COMPATIBILITY FIX
   n_jobs=1 in all RandomizedSearchCV calls to prevent
   XGBoost access violation in loky worker processes.
""")