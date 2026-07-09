from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed. Install requirements.txt first, then rerun this script.")
        return

    mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())
    mlflow.set_experiment("Employee_Attrition_MLOps")

    threshold = float((BASE_DIR / "selected_threshold.txt").read_text().strip())
    metrics_path = BASE_DIR / "artifacts" / "hyperparameter_tuning_results.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    with mlflow.start_run(run_name="milestone_4_deployment_v1"):
        mlflow.log_param("project", "Employee Attrition / Job Change Prediction")
        mlflow.log_param("final_model", "LightGBM Classifier")
        mlflow.log_param("selected_threshold", threshold)
        mlflow.log_param("preprocessing_artifact", "full_pipeline.joblib")
        mlflow.log_param("deployment", "Streamlit + FastAPI")

        if not metrics_df.empty:
            best_row = metrics_df.iloc[0]
            for col in ["Validation Accuracy", "Validation Precision", "Validation Recall", "Validation F1", "Validation ROC-AUC", "Validation PR-AUC"]:
                if col in metrics_df.columns:
                    mlflow.log_metric(col.lower().replace(" ", "_"), float(best_row[col]))

        for artifact in [
            "final_attrition_model.pkl",
            "preprocessor.joblib",
            "full_pipeline.joblib",
            "model_features.csv",
            "selected_threshold.txt",
            "requirements.txt",
            "README.md",
        ]:
            path = BASE_DIR / artifact
            if path.exists():
                mlflow.log_artifact(str(path))

        reports_dir = BASE_DIR / "reports"
        if reports_dir.exists():
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

        print("✅ MLflow run logged under Employee_Attrition_MLOps")


if __name__ == "__main__":
    main()
