"""Feature engineering used by the Employee Attrition deployment pipeline.

This module contains the same transformations used in the final EDA notebook before
fitting the saved ColumnTransformer (`preprocessor.joblib`).  Keeping the function in a
standalone .py file makes the serialized sklearn FunctionTransformer portable for
Streamlit/FastAPI deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS_TO_IMPUTE = [
    "gender",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "company_size",
    "company_type",
    "experience",
    "last_new_job",
]

# Medians learned from the training data in the notebook after mapping string values.
EXPERIENCE_MEDIAN = 9.0
LAST_NEW_JOB_MEDIAN = 1.0
COMPANY_SIZE_MEDIAN = 300.0

SIZE_MAPPING = {
    "<10": 5.0,
    "10-49": 30.0,
    "50-99": 75.0,
    "100-500": 300.0,
    "500-999": 750.0,
    "1000-4999": 3000.0,
    "5000-9999": 7500.0,
    "10000+": 10000.0,
    "Unknown": np.nan,
}

EDUCATION_MAPPING = {
    "Primary School": 1.0,
    "High School": 2.0,
    "Graduate": 3.0,
    "Masters": 4.0,
    "Phd": 5.0,
    "Unknown": 0.0,
}


def _to_float_or_nan(value):
    """Convert a scalar to float; return NaN when conversion is impossible."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def clean_experience(value):
    """Match the notebook mapping for the raw `experience` column."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "Unknown":
        return np.nan
    if value == "<1":
        return 0.5
    if value == ">20":
        return 21.0
    return _to_float_or_nan(value)


def clean_last_new_job(value):
    """Match the notebook mapping for the raw `last_new_job` column."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "Unknown":
        return np.nan
    if value == "never":
        return 0.0
    if value == ">4":
        return 5.0
    return _to_float_or_nan(value)


def apply_feature_engineering(df_in: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact feature engineering used before model-ready preprocessing.

    Parameters
    ----------
    df_in:
        A pandas DataFrame containing raw HR fields from the original dataset/app/API.
        It may also contain extra columns. Extra columns are kept; the saved
        ColumnTransformer selects only the columns it was fitted on.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the engineered intermediate columns required by
        `preprocessor.joblib`.
    """
    df = df_in.copy()

    # Normalise accidental app/user spelling before any mapping.
    if "company_size" in df.columns:
        df["company_size"] = df["company_size"].replace("10/49", "10-49")

    # Same categorical missing-value handling as the notebook.
    for col in CAT_COLS_TO_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Numeric columns from raw text fields, then median-impute using train medians.
    df["experience_numeric"] = df["experience"].apply(clean_experience).fillna(EXPERIENCE_MEDIAN)
    df["last_new_job_numeric"] = df["last_new_job"].apply(clean_last_new_job).fillna(LAST_NEW_JOB_MEDIAN)
    df["company_size_numeric"] = df["company_size"].map(SIZE_MAPPING).fillna(COMPANY_SIZE_MEDIAN)

    # Notebook binning logic. `pd.cut` defaults to right=True, matching the notebook.
    df["experience_group"] = pd.cut(
        df["experience_numeric"],
        bins=[-1, 2, 5, 10, 15, 25],
        labels=["Fresher", "Junior", "Mid", "Senior", "Expert"],
    )
    df["city_development_group"] = pd.cut(
        df["city_development_index"],
        bins=[0, 0.6, 0.75, 0.85, 1.0],
        labels=["Low", "Medium", "High", "Very High"],
    )
    df["training_hours_group"] = pd.cut(
        df["training_hours"],
        bins=[0, 23, 47, 88, 400],
        labels=["Low", "Medium", "High", "Very High"],
    )

    # Logical features from the notebook.
    df["has_relevant_experience_binary"] = (
        df["relevent_experience"] == "Has relevent experience"
    ).astype(int)
    df["is_currently_enrolled"] = df["enrolled_university"].isin(
        ["Full time course", "Part time course"]
    ).astype(int)
    df["company_known_flag"] = (df["company_size"] != "Unknown").astype(int)
    df["education_rank"] = df["education_level"].map(EDUCATION_MAPPING).fillna(0).astype(int)

    # Ratios/interactions.
    df["job_mobility_score"] = df["experience_numeric"] / (df["last_new_job_numeric"] + 1)
    df["experience_to_training_ratio"] = df["training_hours"] / (df["experience_numeric"] + 1)
    df["city_experience_interaction"] = df["city_development_index"] * df["experience_numeric"]

    # Final experimental flags from Milestone 2 Section 17.
    df["low_city_development_flag"] = (df["city_development_index"] < 0.6).astype(int)
    df["high_training_flag"] = (df["training_hours"] > 100).astype(int)
    df["senior_experience_flag"] = (df["experience_numeric"] >= 10).astype(int)
    df["unknown_company_info_flag"] = (df["company_known_flag"] == 0).astype(int)

    return df
