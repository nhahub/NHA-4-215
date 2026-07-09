# Employee Attrition Prediction — Milestone 4 Deployment

This folder contains the final Milestone 4 deployment for the Employee Attrition / Job Change Prediction project.

## What this deployment uses

- `final_attrition_model.pkl`: final LightGBM model from Milestone 3.
- `preprocessor.joblib`: fitted ColumnTransformer from Milestone 2.
- `feature_engineering.py`: exact feature engineering from the final EDA notebook.
- `full_pipeline.joblib`: unified deployment preprocessing pipeline created from `feature_engineering.py` + `preprocessor.joblib`.
- `selected_threshold.txt`: selected classification threshold.
- `model_features.csv`: exact model feature order.

The Streamlit app and FastAPI API do not manually scale or one-hot encode the input. They call `full_pipeline.joblib`.

## Folder structure

```text
employee_attrition_milestone_4_final/
├── app.py
├── api.py
├── feature_engineering.py
├── create_pipeline.py
├── preprocessing.py
├── validate_deployment.py
├── mlops_tracking.py
├── final_attrition_model.pkl
├── preprocessor.joblib
├── full_pipeline.joblib
├── model_features.csv
├── selected_threshold.txt
├── prediction_logs.csv
├── requirements.txt
├── data/
└── reports/
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Create or recreate the full pipeline

```bash
python create_pipeline.py
```

## Validate deployment correctness

Run this before using the app/API:

```bash
python validate_deployment.py
```

Expected result:

```text
✅ PASS: raw input → full_pipeline.joblib matches model_ready_train.csv
✅ PASS: columns, order, scaling, encoding, and values match
✅ PASS: model predictions match exactly
```

## Run Streamlit dashboard

```bash
python -m streamlit run app.py
```

## Run FastAPI API

```bash
uvicorn api:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Example request body for `/predict`:

```json
{
  "city_development_index": 0.624,
  "gender": "Male",
  "relevent_experience": "No relevent experience",
  "enrolled_university": "Full time course",
  "education_level": "Graduate",
  "major_discipline": "STEM",
  "experience": "<1",
  "company_size": "Unknown",
  "company_type": "Unknown",
  "last_new_job": "never",
  "training_hours": 120
}
```

## Monitoring

Every Streamlit/API prediction is appended to `prediction_logs.csv`. The Streamlit Monitoring tab displays:

- Total predictions
- Average predicted probability
- High-risk percentage
- Class 1 prediction percentage
- Risk-level distribution
- Probability trend over time
- Alerts when high-risk share or average probability becomes unusually high

## MLflow tracking

```bash
python mlops_tracking.py
```

MLflow logs the model artifacts, preprocessing artifacts, threshold, validation metrics, and reports to a local `mlruns/` directory.
