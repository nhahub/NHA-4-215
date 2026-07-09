# Retraining Strategy

## Why retraining is needed

Employee behavior and labor-market conditions can change over time. A model trained on old patterns may become less accurate when workforce behavior, hiring conditions, or training programs change.

## Recommended retraining schedule

Retrain the model every 3 to 6 months, depending on the availability of new labeled data.

## Trigger-based retraining

Retraining should be triggered earlier if any of the following occurs:

- Significant data drift in key input features.
- Sudden increase in high-risk predictions.
- Decrease in model performance after actual outcomes become available.
- Increase in false negatives for attrition-risk employees.
- Major business or workforce policy changes.

## Retraining pipeline

1. Collect new HR/candidate data.
2. Validate schema and data quality.
3. Apply the same feature-engineering and preprocessing pipeline.
4. Train candidate models.
5. Compare against the current deployed model.
6. Select the best model based on F1, Recall, ROC-AUC, and PR-AUC.
7. Validate the selected threshold.
8. Save new model and preprocessing artifacts.
9. Run `validate_deployment.py`.
10. Deploy the new model version after human review.

## Model versioning

Each new deployment should have a version label, for example:

```text
v1_lightgbm_milestone3
v2_lightgbm_retrained_2026Q4
```

The deployed model should not be replaced unless validation and business review are completed.
