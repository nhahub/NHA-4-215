"""Build the full deployment preprocessing pipeline.

Run once after placing `preprocessor.joblib` and `feature_engineering.py` in this folder.
The output `full_pipeline.joblib` is the only preprocessing artifact used by app.py/api.py.
"""
from __future__ import annotations

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_engineering import apply_feature_engineering

PREPROCESSOR_PATH = "preprocessor.joblib"
OUTPUT_PATH = "full_pipeline.joblib"


def main() -> None:
    print("Building full deployment pipeline...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_engineering = FunctionTransformer(apply_feature_engineering, validate=False)

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("scaling_and_encoding", preprocessor),
        ]
    )
    joblib.dump(full_pipeline, OUTPUT_PATH)
    print(f"✅ Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
