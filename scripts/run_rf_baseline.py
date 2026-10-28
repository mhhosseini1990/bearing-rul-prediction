"""Minimal *real-data* RF baseline; not a production system.

Usage: python scripts/run_rf_baseline.py --data-dir data/raw/XJTU-SY
"""
import argparse
import json
from pathlib import Path

import joblib

from bearing_rul.data.xjtu_loader import list_measurements
from bearing_rul.data.rul_labels import add_remaining_recordings
from bearing_rul.data.splits import split_by_bearing
from bearing_rul.evaluation.metrics import regression_metrics
from bearing_rul.evaluation.replay import historical_replay
from bearing_rul.features.time_domain import feature_table
from bearing_rul.models.random_forest import fit_random_forest, TARGET


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate simple bearing RUL baseline")
    parser.add_argument("--data-dir", default="data/raw/XJTU-SY")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    manifest = list_measurements(args.data_dir)
    print(f"Loaded file list: {len(manifest)} CSV files; {manifest['bearing_id'].nunique()} bearings")
    print("Extracting features. Large source datasets may need significant local resources.")
    table = add_remaining_recordings(feature_table(manifest))
    train, test = split_by_bearing(table)

    model = fit_random_forest(train)
    replay = historical_replay(model, test)
    overall = regression_metrics(replay[TARGET], replay["predicted_rul"])
    per_bearing = {
        bearing: regression_metrics(group[TARGET], group["predicted_rul"])
        for bearing, group in replay.groupby("bearing_id")
    }
    summary = {
        "rul_unit": "remaining_recordings",
        "model": "RandomForestRegressor",
        "train_bearings": sorted(train["bearing_id"].unique().tolist()),
        "test_bearings": sorted(test["bearing_id"].unique().tolist()),
        "overall": overall,
        "per_bearing": per_bearing,
        "caution": "Baseline only; check source protocol and tune using train/validation bearings.",
    }

    output_root = Path(args.output_dir)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "models").mkdir(parents=True, exist_ok=True)
    replay.to_csv(output_root / "metrics" / "rf_replay_predictions.csv", index=False)
    (output_root / "metrics" / "rf_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(model, output_root / "models" / "random_forest.joblib")
    print("RF baseline MAE/RMSE (remaining recordings):", overall)
    print("Results saved locally under:", output_root)


if __name__ == "__main__":
    main()
