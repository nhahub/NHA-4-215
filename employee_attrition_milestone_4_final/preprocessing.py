"""Thin deployment wrapper around the saved full preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "full_pipeline.joblib"
FEATURES_PATH = BASE_DIR / "model_features.csv"

RAW_COLUMNS = [
    "city_development_index",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
    "training_hours",
]


def _load_expected_features() -> list[str]:
    features_df = pd.read_csv(FEATURES_PATH)
    if "features" in features_df.columns:
        cols = features_df["features"].tolist()
    else:
        cols = features_df.iloc[:, 0].tolist()
    return [c for c in cols if c != "target"]


expected_model_cols = _load_expected_features()
pipeline = joblib.load(PIPELINE_PATH)


def prepare_dataframe_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transform raw employee rows into the model-ready feature matrix.

    This function does not manually encode/scale anything. It only calls the saved
    `full_pipeline.joblib` and validates the output shape/column order.
    """
    missing = [col for col in RAW_COLUMNS if col not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required raw input columns: {missing}")

    # Keep raw columns only to mimic the Streamlit/API input contract.
    df_input = df_raw[RAW_COLUMNS].copy()
    transformed = pipeline.transform(df_input)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if transformed.shape[1] != len(expected_model_cols):
        raise ValueError(
            f"Pipeline output has {transformed.shape[1]} columns, "
            f"but model_features.csv expects {len(expected_model_cols)}."
        )

    return pd.DataFrame(transformed, columns=expected_model_cols, index=df_raw.index)


def prepare_input_for_model(input_data: Mapping[str, Any]) -> pd.DataFrame:
    """Transform a single raw employee dictionary into model-ready features."""
    return prepare_dataframe_for_model(pd.DataFrame([dict(input_data)]))


def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"
