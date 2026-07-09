# MLOps Report — Employee Attrition Prediction

## Project overview

This project predicts employee attrition / job-change risk using cleaned HR candidate data and a final LightGBM classifier selected in Milestone 3.

## Milestone 4 objective

The goal of Milestone 4 is to move the final model from notebook experimentation into a reproducible deployment layer with monitoring and retraining planning.

## Final model and artifacts

The deployment uses these saved artifacts:

- `final_attrition_model.pkl`: final LightGBM model.
- `preprocessor.joblib`: fitted sklearn ColumnTransformer from Milestone 2.
- `feature_engineering.py`: notebook feature-engineering logic converted into a reusable module.
- `full_pipeline.joblib`: unified pipeline combining feature engineering and preprocessing.
- `model_features.csv`: exact final feature list and column order.
- `selected_threshold.txt`: classification threshold.
- `prediction_logs.csv`: production prediction log.

## Pipeline design

The production flow is:

```text
Raw HR input
→ full_pipeline.joblib
→ model-ready 41 features
→ final_attrition_model.pkl
→ selected_threshold.txt
→ predicted class and risk level
```

This prevents the app/API from manually recreating scaling or one-hot encoding. The deployment wrapper only loads fitted artifacts and calls `.transform()`.

## Experiment tracking

The project includes `mlops_tracking.py`, which uses MLflow to log:

- final model name
- selected threshold
- validation metrics from Milestone 3
- model artifact
- preprocessing artifacts
- reports

The MLflow experiment name is `Employee_Attrition_MLOps`.

## Deployment architecture

Two deployment interfaces are provided:

1. **Streamlit dashboard** through `app.py` for HR users.
2. **FastAPI service** through `api.py` for real-time API predictions.

Both interfaces call the same `preprocessing.py` wrapper and therefore produce consistent predictions.

## Reproducibility controls

- The preprocessing pipeline is serialized.
- The model is serialized.
- The final feature order is stored in `model_features.csv`.
- `validate_deployment.py` compares deployment pipeline output against `model_ready_train.csv`.
- Package dependencies are listed in `requirements.txt`.

## Limitations

The dataset lacks direct workplace factors such as salary, performance rating, job satisfaction, manager relationship, or work-life balance. Therefore, the model relies on proxy variables such as city development, experience, company information, and training behavior.
