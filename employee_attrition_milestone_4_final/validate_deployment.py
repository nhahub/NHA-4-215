from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import RAW_COLUMNS, expected_model_cols, prepare_dataframe_for_model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "final_attrition_model.pkl"


def _report_column_differences(deployment_cols: list[str], expected_cols: list[str]) -> None:
    dep_set = set(deployment_cols)
    exp_set = set(expected_cols)
    print("Missing in deployment:", sorted(exp_set - dep_set))
    print("Extra in deployment:", sorted(dep_set - exp_set))


def validate(sample_size: int = 250) -> bool:
    print("Executing strict Milestone 4 validation...\n")
    cleaned = pd.read_csv(DATA_DIR / "cleaned_train.csv")
    ready = pd.read_csv(DATA_DIR / "model_ready_train.csv")
    model = joblib.load(MODEL_PATH)

    sample_raw = cleaned[RAW_COLUMNS].head(sample_size).copy()
    expected_ready = ready.head(sample_size).copy()
    if "target" in expected_ready.columns:
        expected_ready = expected_ready.drop(columns=["target"])

    deployment_ready = prepare_dataframe_for_model(sample_raw)

    failures: list[str] = []

    if deployment_ready.shape != expected_ready.shape:
        failures.append(f"Shape mismatch: deployment {deployment_ready.shape} vs expected {expected_ready.shape}")

    if list(deployment_ready.columns) != list(expected_ready.columns):
        failures.append("Column names/order mismatch.")
        _report_column_differences(list(deployment_ready.columns), list(expected_ready.columns))

    if deployment_ready.isna().sum().sum() != 0:
        failures.append("Deployment output contains missing values.")

    if not np.allclose(deployment_ready.values.astype(float), expected_ready.values.astype(float), atol=1e-6):
        abs_diff = np.abs(deployment_ready.values.astype(float) - expected_ready.values.astype(float))
        max_diff = abs_diff.max()
        failures.append(f"Numeric values mismatch. Maximum absolute difference = {max_diff:.8f}")

        diff_by_col = pd.Series(abs_diff.max(axis=0), index=deployment_ready.columns).sort_values(ascending=False)
        print("Top different columns:")
        print(diff_by_col.head(15))

    # Prediction equivalence check.
    pred_from_deployment = model.predict_proba(deployment_ready)[:, 1]
    pred_from_ready = model.predict_proba(expected_ready[expected_model_cols])[:, 1]
    if not np.allclose(pred_from_deployment, pred_from_ready, atol=1e-10):
        failures.append("Predictions from deployment pipeline do not match predictions from model_ready_train rows.")

    if failures:
        print("🚨 VALIDATION FAILED 🚨")
        for failure in failures:
            print("-", failure)
        return False

    print("✅ PASS: raw input → full_pipeline.joblib matches model_ready_train.csv")
    print("✅ PASS: columns, order, scaling, encoding, and values match")
    print("✅ PASS: model predictions match exactly")
    print("\nDeployment is ready.")
    return True


if __name__ == "__main__":
    ok = validate()
    raise SystemExit(0 if ok else 1)
