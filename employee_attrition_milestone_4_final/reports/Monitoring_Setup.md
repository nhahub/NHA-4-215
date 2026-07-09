# Monitoring Setup

## Purpose

Monitoring ensures the deployed attrition prediction system remains reliable after deployment.

## Logged predictions

Every prediction from Streamlit or FastAPI is appended to `prediction_logs.csv` with:

- timestamp
- model version
- predicted probability
- selected threshold
- predicted class
- risk level
- raw input values

## Metrics tracked

The Streamlit Monitoring tab tracks:

- Total prediction count
- Average predicted probability
- Percentage of high-risk predictions
- Percentage of predicted class 1
- Risk-level distribution
- Average predicted probability over time

## Alert rules

Alerts are raised when:

- High-risk predictions exceed 50% of logged predictions.
- Average predicted probability exceeds 70%.

Additional recommended monitoring checks:

- Missing value rates in new inputs.
- Distribution drift in city development index, training hours, and experience.
- Sudden increase in predicted attrition risk.
- Model performance degradation when actual outcomes become available.

## Monitoring frequency

- Prediction logs should be reviewed daily during active use.
- Drift and performance reports should be reviewed weekly or monthly depending on data volume.

## Action when alerts occur

1. Review recent input records.
2. Check whether the change reflects real business conditions or data quality issues.
3. Compare new input distributions with training data.
4. If drift is confirmed, start the retraining process.
