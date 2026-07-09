from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import expected_model_cols, get_risk_level, prepare_input_for_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_attrition_model.pkl"
THRESHOLD_PATH = BASE_DIR / "selected_threshold.txt"
LOGS_PATH = BASE_DIR / "prediction_logs.csv"

model = joblib.load(MODEL_PATH)
threshold = float(THRESHOLD_PATH.read_text().strip())

app = FastAPI(title="Employee Attrition Prediction API", version="1.0.0")


class EmployeeInput(BaseModel):
    city_development_index: float = Field(..., ge=0.0, le=1.0)
    gender: str
    relevent_experience: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: str
    company_size: str
    company_type: str
    last_new_job: str
    training_hours: float = Field(..., ge=0.0)


@app.get("/")
def root():
    return {
        "message": "Employee Attrition Prediction API is running.",
        "model": "final_attrition_model.pkl",
        "pipeline": "full_pipeline.joblib",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": threshold,
        "feature_count": len(expected_model_cols),
    }


@app.post("/predict")
def predict(employee: EmployeeInput):
    try:
        raw_dict = employee.model_dump() if hasattr(employee, "model_dump") else employee.dict()
        processed_df = prepare_input_for_model(raw_dict)
        probability = float(model.predict_proba(processed_df)[0][1])
        predicted_class = int(probability >= threshold)
        risk_level = get_risk_level(probability)

        log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": "v1_lightgbm_milestone3",
            "predicted_probability": round(probability, 6),
            "selected_threshold": threshold,
            "predicted_class": predicted_class,
            "risk_level": risk_level,
        }
        log_row.update(raw_dict)
        pd.DataFrame([log_row]).to_csv(
            LOGS_PATH,
            mode="a",
            header=not LOGS_PATH.exists(),
            index=False,
        )

        return {
            "predicted_probability": probability,
            "selected_threshold": threshold,
            "predicted_class": predicted_class,
            "risk_level": risk_level,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
