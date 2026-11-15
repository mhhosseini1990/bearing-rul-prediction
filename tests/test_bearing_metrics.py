import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bearing_rul.evaluation.bearing_metrics import (
    build_inspectable_metric_report,
    evaluate_per_bearing_minutes,
    recompute_metrics_from_saved_predictions,
    verify_identical_prediction_targets,
)


def test_unequal_bearing_lengths_and_inspectable_weights():
    """Verify unequal bearing lengths expose exact window weights and metric divergence."""
    df_long = pd.DataFrame(
        {
            "bearing_id": ["Bearing_Long"] * 90,
            "actual_rul": [100.0] * 90,
            "predicted_rul": [102.0] * 90,
        }
    )
    df_short = pd.DataFrame(
        {
            "bearing_id": ["Bearing_Short"] * 10,
            "actual_rul": [50.0] * 10,
            "predicted_rul": [72.0] * 10,
        }
    )
    df = pd.concat([df_long, df_short], ignore_index=True)

    per_df = evaluate_per_bearing_minutes(df)
    long_row = per_df[per_df["bearing_id"] == "Bearing_Long"].iloc[0]
    short_row = per_df[per_df["bearing_id"] == "Bearing_Short"].iloc[0]

    assert long_row["window_count"] == 90
    assert short_row["window_count"] == 10
    assert pytest.approx(long_row["pooled_window_weight"], rel=1e-6) == 0.90
    assert pytest.approx(short_row["pooled_window_weight"], rel=1e-6) == 0.10
    assert pytest.approx(long_row["equal_bearing_weight"], rel=1e-6) == 0.50
    assert pytest.approx(short_row["equal_bearing_weight"], rel=1e-6) == 0.50
    assert pytest.approx(short_row["weight_delta"], rel=1e-6) == 0.40

    report = build_inspectable_metric_report(df, model_name="lstm")
    assert report["pooled_window_weighted"]["aggregation_method"] == "pooled_window_weighted"
    assert pytest.approx(report["pooled_window_weighted"]["mae_minutes"], rel=1e-6) == 4.0

    assert report["equal_bearing_mean"]["aggregation_method"] == "equal_bearing_mean"
    assert pytest.approx(report["equal_bearing_mean"]["mae_minutes"], rel=1e-6) == 12.0
    assert pytest.approx(report["aggregation_gap"]["mae_minutes_diff"], rel=1e-6) == 8.0


def test_independent_recomputation_from_saved_predictions(tmp_path: Path):
    saved_csv = tmp_path / "predictions_lstm.csv"
    df_saved = pd.DataFrame(
        {
            "bearing_id": ["B1", "B1", "B2"],
            "actual_rul": [30.0, 20.0, 40.0],
            "predicted_rul": [33.0, 17.0, 46.0],
        }
    )
    df_saved.to_csv(saved_csv, index=False)

    out_dir = tmp_path / "recomputed_metrics"
    report = recompute_metrics_from_saved_predictions(
        predictions_csv_path=saved_csv,
        model_name="lstm",
        output_dir=out_dir,
    )

    per_csv = out_dir / "lstm_per_bearing_metrics.csv"
    summary_json = out_dir / "lstm_metric_summary.json"
    assert per_csv.exists()
    assert summary_json.exists()

    exported_df = pd.read_csv(per_csv)
    assert set(["bearing_id", "window_count", "mae_minutes", "rmse_minutes"]).issubset(
        exported_df.columns
    )
    assert pytest.approx(report["pooled_window_weighted"]["mae_minutes"], rel=1e-6) == 4.0
    assert pytest.approx(report["equal_bearing_mean"]["mae_minutes"], rel=1e-6) == 4.5

    loaded_json = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded_json["equal_bearing_mean"]["n_bearings"] == 2


def test_verify_identical_prediction_targets_for_lstm_and_gru():
    lstm_df = pd.DataFrame(
        {
            "bearing_id": ["B1", "B1", "B2"],
            "window_idx": [0, 1, 0],
            "actual_rul": [120.0, 119.0, 85.0],
            "predicted_rul": [118.0, 117.5, 88.0],
        }
    )
    gru_df_valid = pd.DataFrame(
        {
            "bearing_id": ["B1", "B1", "B2"],
            "window_idx": [0, 1, 0],
            "actual_rul": [120.0, 119.0, 85.0],
            "predicted_rul": [121.0, 120.0, 82.0],
        }
    )

    verification = verify_identical_prediction_targets(
        lstm_df, gru_df_valid, window_col="window_idx"
    )
    assert verification["targets_identical"] is True
    assert verification["per_bearing_window_counts"] == {"B1": 2, "B2": 1}

    gru_df_mismatch = gru_df_valid.copy()
    gru_df_mismatch.loc[2, "actual_rul"] = 90.0

    with pytest.raises(ValueError, match="targets differ between LSTM and GRU"):
        verify_identical_prediction_targets(lstm_df, gru_df_mismatch, window_col="window_idx")