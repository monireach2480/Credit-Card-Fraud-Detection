# Credit Card Fraud Detection

A machine learning system that detects fraudulent credit card transactions using XGBoost with SHAP explainability, deployed via FastAPI and Streamlit.

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── data/
│   ├── raw/               # Original dataset (creditcard.csv)
│   └── processed/         # Train/test splits, scaled features, SMOTE output
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