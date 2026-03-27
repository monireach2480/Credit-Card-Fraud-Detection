"""
FastAPI Backend — Credit Card Fraud Detection
Loads trained artifacts from deployment_artifacts/ and exposes a /predict endpoint.
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "deployment_artifacts")

# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "final_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "threshold.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(
        f"Missing artifact: {e}. Make sure deployment_artifacts/ exists and "
        "notebook 06_final_pipeline_export.ipynb has been run."
    )

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict whether a credit card transaction is fraudulent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    """Input schema: Time, V1-V28, Amount."""
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0,
                "V1": -2.312, "V2": 1.952, "V3": -1.610, "V4": 3.998,
                "V5": -0.522, "V6": -1.427, "V7": -2.537, "V8": 1.392,
                "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.900,
                "V13": -0.595, "V14": -4.289, "V15": 0.390, "V16": -1.141,
                "V17": -2.831, "V18": -0.017, "V19": 0.416, "V20": 0.126,
                "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320,
                "V25": 0.045, "V26": 0.177, "V27": 0.261, "V28": -0.143,
                "Amount": 1.0
            }
        }


class PredictionOutput(BaseModel):
    prediction: int          # 0 = not fraud, 1 = fraud
    label: str               # "Fraud" or "Not Fraud"
    fraud_probability: float
    threshold: float
    features_used: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Credit Card Fraud Detection API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": float(threshold),
        "features": len(feature_columns),
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput) -> Dict[str, Any]:
    """
    Predict if a transaction is fraudulent.

    Returns prediction (0/1), fraud probability, and the label.
    """
    try:
        # Convert input to DataFrame and reorder columns
        input_dict = transaction.model_dump()
        df = pd.DataFrame([input_dict])
        df = df[feature_columns]

        # Scale features
        X_scaled = scaler.transform(df)

        # Get fraud probability and apply threshold
        fraud_prob = float(model.predict_proba(X_scaled)[0][1])
        prediction = int(fraud_prob >= threshold)
        label = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": prediction,
            "label": label,
            "fraud_probability": round(fraud_prob, 6),
            "threshold": float(threshold),
            "features_used": len(feature_columns),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionInput]):
    """Predict on multiple transactions at once."""
    try:
        rows = [t.model_dump() for t in transactions]
        df = pd.DataFrame(rows)[feature_columns]
        X_scaled = scaler.transform(df)
        probs = model.predict_proba(X_scaled)[:, 1]
        predictions = (probs >= threshold).astype(int)

        return {
            "count": len(transactions),
            "results": [
                {
                    "index": i,
                    "prediction": int(predictions[i]),
                    "label": "Fraud" if predictions[i] == 1 else "Not Fraud",
                    "fraud_probability": round(float(probs[i]), 6),
                }
                for i in range(len(transactions))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
