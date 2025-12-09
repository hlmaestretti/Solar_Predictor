"""
Compares the training feature data with the current production
prediction logs using whylogs v1.6.4 drift metrics.

This script:
  - loads data/processed/features.parquet   (reference/training)
  - loads app/prediction_logs.csv           (current/production)
  - logs both with whylogs
  - computes per-column drift scores using calculate_drift_scores
  - writes monitoring/drift_report.json
"""

from pathlib import Path
import json

import pandas as pd
import whylogs as why
from whylogs.viz.drift.column_drift_algorithms import calculate_drift_scores  # v1.6.4 API

REF_PATH = Path("data/processed/features.parquet")
LOG_PATH = Path("app/prediction_logs.csv")
OUTPUT_REPORT = Path("monitoring/drift_report.json")

# Must match how you wrote the CSV in FastAPI:
LOG_COLUMNS = [
    "timestamp",
    "hour",
    "dayofyear",
    "dayofweek",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "prediction",
]


def main():
    if not REF_PATH.exists():
        raise FileNotFoundError(f"Reference features not found at: {REF_PATH}")
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction logs not found at: {LOG_PATH}")

    print("Loading reference features...")
    ref_df = pd.read_parquet(REF_PATH)

    print("Loading production logs...")
    curr_df = pd.read_csv(LOG_PATH, names=LOG_COLUMNS)

    # Drop timestamp; we only care about feature distributions:
    curr_df = curr_df.drop(columns=["timestamp"])

    # Align columns: use intersection of both (in case of differences)
    common_cols = [c for c in ref_df.columns if c in curr_df.columns]
    ref_df = ref_df[common_cols]
    curr_df = curr_df[common_cols]

    print(f"Columns used for drift analysis: {common_cols}")

    print("Logging reference dataset with whylogs...")
    ref_view = why.log(ref_df).view()

    print("Logging current dataset with whylogs...")
    curr_view = why.log(curr_df).view()

    print("Calculating drift scores...")
    scores = calculate_drift_scores(
        target_view=curr_view,
        reference_view=ref_view,
        with_thresholds=True,
    )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(scores, f, indent=4)

    print(f"Drift report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
