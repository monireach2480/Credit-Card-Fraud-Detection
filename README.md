# Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions using XGBoost with SHAP explainability, deployed via FastAPI and Streamlit.

---


## 📌 Project Overview

Credit card fraud is a significant issue in financial systems, leading to substantial financial losses and security concerns. Detecting fraudulent transactions is challenging due to the highly imbalanced nature of the data, where fraud cases represent only a tiny fraction of all transactions.

This project builds a machine learning pipeline to detect fraudulent transactions using advanced models, with a focus on handling class imbalance and improving detection performance while maintaining interpretability.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions using machine learning
- Handle severe class imbalance effectively
- Compare baseline and advanced models
- Improve detection performance using XGBoost
- Provide explainability using SHAP
- Build a deployable prediction pipeline

---

## 📂 Project Structure
```
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_preprocessing_imbalance.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_xgboost_tuning.ipynb
│   ├── 05_evaluation_explainability.ipynb
│   └── 06_final_pipeline_export.ipynb
├── src/
│   ├── data_preprocessing.py   # Load, split, scale, SMOTE
│   ├── train_model.py          # LR, RF, XGBoost training
│   ├── evaluate.py             # Metrics, ROC/PR curves, threshold tuning
│   ├── explain.py              # SHAP explanations
│   └── utils.py                # Shared utilities
├── models/
│   ├── baseline/          # logistic_regression, random_forest .pkl files
│   └── final/             # best_xgboost_model.pkl
├── deployment_artifacts/  # final_model.pkl, scaler.pkl, threshold.pkl, schemas
├── reports/
│   ├── figures/           # Saved plots (ROC, confusion matrix, SHAP)
│   └── tables/            # CSV result tables
├── api/
│   └── main.py            # FastAPI backend
├── app/
│   └── streamlit_app.py   # Streamlit frontend
├── requirements.txt
└── .gitignore
```

---

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Precision | 0.88   |
| Recall    | 0.84   |
| F1        | 0.86   |
| ROC-AUC   | 0.97   |
| PR-AUC    | 0.88   |

**Final model:** XGBoost with `scale_pos_weight ≈ 577`  
**Key features:** V14, V4, V12 (SHAP analysis)

---

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+

---

## Run the Project

### 1. Explore Notebooks
```bash
jupyter notebook
```
Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

### 2. Start the API
```bash
uvicorn api.main:app --reload
```
API docs available at: http://127.0.0.1:8000/docs

### 3. Start the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
> Make sure the API is running first.

---

## API Endpoints

| Method | Endpoint        | Description                   |
|--------|-----------------|-------------------------------|
| GET    | `/health`       | Check API status              |
| POST   | `/predict`      | Predict a single transaction  |
| POST   | `/predict/batch`| Predict multiple transactions |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @deployment_artifacts/sample_request.json
```

---

## Team Division

| Member | Notebooks |
|--------|-----------|
| EDA | 01_data_understanding_eda |
| Preprocessing | 02_preprocessing_imbalance |
| Modeling Lead | 03, 04, 05 |
| Deployment | 06, api/main.py, app/streamlit_app.py |

---

## Key Technical Decisions

- **Metric priority:** Recall > Precision (missing fraud is more costly than a false alert)
- **Imbalance handling:** `scale_pos_weight` (more stable than SMOTE for tree models)  
- **Explainability:** SHAP TreeExplainer for global + local feature attribution
- **Avoid accuracy:** Dataset is ~99.8% non-fraud; accuracy is misleading
│
├── models/
│   ├── baseline/
│   └── final/
│
├── reports/
│   └── tables/
│
├── deployment_artifacts/
│
├── api/                # FastAPI backend (to be implemented)
├── app/                # Streamlit frontend (to be implemented)
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Credit card transactions dataset |
| **Features** | `Time`, `Amount`, and anonymized features `V1–V28` |
| **Target** | `Class` → `0` = Non-Fraud, `1` = Fraud |
| **Challenge** | Highly imbalanced dataset (fraud cases are very rare) |

---

## ⚙️ Methodology

### 1. Data Understanding & EDA
- Checked data structure and missing values
- Analyzed class imbalance
- Explored feature distributions
- Compared fraud vs. non-fraud patterns

### 2. Data Preprocessing
- Train-test split (stratified)
- Feature scaling using `StandardScaler`
- Imbalance handling via **SMOTE** (applied only on training data)
- Saved: scaled datasets, resampled datasets, raw datasets for deployment

### 3. Baseline Models
- Logistic Regression
- Random Forest

Evaluated using: Precision, Recall, F1-score, ROC-AUC, PR-AUC

### 4. Advanced Modeling — XGBoost
Implemented multiple approaches:
- Initial XGBoost model
- XGBoost with `scale_pos_weight`
- XGBoost trained on SMOTE data
- Hyperparameter tuning via `RandomizedSearchCV`

Key tuning parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`

### 5. Model Selection

> 👉 **Final Model: XGBoost with `scale_pos_weight`**

Selected for its best balance between precision, recall, F1-score, and PR-AUC — making it most suitable for real-world fraud detection.

### 6. Evaluation

| Metric | Score |
|--------|-------|
| Precision | ~0.88 |
| Recall | ~0.84 |
| F1-Score | ~0.86 |
| ROC-AUC | ~0.97 |
| PR-AUC | ~0.88 |

Visualizations: Confusion Matrix · ROC Curve · Precision-Recall Curve

### 7. Explainability — SHAP
- Global feature importance
- Local explanations for individual predictions
- Key features identified: `V14`, `V4`, `V12`

### 8. Prediction Pipeline
An end-to-end prediction function that:
- Accepts raw transaction input
- Applies preprocessing (scaling)
- Predicts fraud probability
- Applies classification threshold
- Returns prediction results

---

## 🚀 Deployment (Planned)

### Backend — FastAPI
- Load model and preprocessing artifacts
- Provide `/predict` API endpoint
- Return prediction and probability

### Frontend — Streamlit
- User input interface
- Display fraud prediction results
- Communicate with FastAPI backend

---

## 📦 Deployment Artifacts

Saved for integration:
```
deployment_artifacts/
├── final_model.pkl
├── scaler.pkl
├── feature_columns.pkl
└── threshold.pkl
```

---

## 🛠️ Installation
```bash
git clone <your-repo-link>
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run Notebooks**
```bash
jupyter notebook notebooks/
```

**Test Prediction Pipeline**
```bash
# Open and run:
notebooks/06_final_pipeline_export.ipynb
```

---

## 📈 Key Insights

- Class imbalance is the main challenge in fraud detection
- XGBoost performs best on structured/tabular data
- Handling imbalance is critical for meaningful performance
- Recall and PR-AUC matter more than raw accuracy
- SHAP explainability improves trust and real-world usability



---

## 📄 License

This project is developed for **academic purposes only**.
