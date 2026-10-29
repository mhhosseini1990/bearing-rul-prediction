"""Unit and CLI integration tests using *synthetic* predictions only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bearing_rul.simulation.maintenance_replay import (  # noqa: E402
    ReplaySettings, first_alert_index, load_prediction_files,
    normalize_predictions, replay_bearing, run_replay,
)


def _raw(bearing: str, predictions: list[float], actual: list[float] | None = None) -> pd.DataFrame:
    n = len(predictions)
    return pd.DataFrame({
        "bearing_id": [bearing] * n,
        "target_step": list(range(n)),
        "actual_rul_minutes": actual if actual is not None else list(range(n - 1, -1, -1)),
        "predicted_rul_minutes": predictions,
    })


def test_first_alert_requires_consecutive_low_predictions():
    settings = ReplaySettings(120, 3, 30)
    assert first_alert_index([140, 110, 150, 100, 90, 70], settings) == 5
    assert first_alert_index([140, 110, 150, 100, 90], settings) is None


def test_online_policy_is_unchanged_by_future_predictions():
    settings = ReplaySettings(120, 2, 30)
    prefix = [200, 110, 100]
    a = first_alert_index(prefix + [0, 0], settings)
    b = first_alert_index(prefix + [1000, 1000], settings)
    assert a == b == 2


@pytest.mark.parametrize("prediction,expected_status,lead", [
    ([100, 90, 80, 70, 60], "early", 270),
    ([250, 220, 110, 100, 90], "timely", 60),
    ([250, 220, 160, 100, 90], "late", 0),
    ([250, 220, 160, 150, 140], "missed", None),
])
def test_all_event_statuses(prediction, expected_status, lead):
    actual_rul = {
        "early": [300, 270, 240, 30, 0],
        "timely": [240, 180, 120, 60, 0],
        "late": [100, 80, 50, 15, 0],
        "missed": [100, 80, 50, 15, 0],
    }[expected_status]
    data = normalize_predictions(_raw("Bearing1_5", prediction, actual_rul), "GRU")
    event, trace = replay_bearing(data, ReplaySettings(120, 2, 30))
    assert event["status"] == expected_status
    assert event["actual_lead_time_minutes"] == lead
    assert int(trace["alert_issued"].sum()) == int(lead is not None)


def test_missing_chronology_rejected():
    data = _raw("Bearing1_5", [50, 30, 20]).drop(columns="target_step")
    with pytest.raises(ValueError, match="Missing 'step'"):
        normalize_predictions(data, "GRU")


def test_duplicate_step_rejected():
    data = _raw("Bearing1_5", [50, 30, 20]); data.loc[1, "target_step"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        normalize_predictions(data, "GRU")


def test_nonfinite_prediction_rejected():
    data = _raw("Bearing1_5", [50, float("nan"), 20])
    with pytest.raises(ValueError, match="finite"):
        normalize_predictions(data, "GRU")


def test_incomplete_failure_history_rejected():
    data = _raw("Bearing1_5", [50, 30, 20], [30, 20, 10])
    with pytest.raises(ValueError, match="last actual RUL"):
        normalize_predictions(data, "GRU")


def test_actual_rul_is_not_used_to_choose_alert():
    settings = ReplaySettings(120, 2, 20)
    x = normalize_predictions(_raw("B", [300, 100, 90, 50], [300, 200, 50, 0]), "LSTM")
    y = normalize_predictions(_raw("B", [300, 100, 90, 50], [20, 15, 5, 0]), "LSTM")
    ex, _ = replay_bearing(x, settings)
    ey, _ = replay_bearing(y, settings)
    assert ex["alert_step"] == ey["alert_step"] == 2
    assert ex["actual_lead_time_minutes"] != ey["actual_lead_time_minutes"]


def test_both_models_have_identical_prediction_alignment(tmp_path):
    left = _raw("B", [50, 30, 20]); right = _raw("B", [60, 50, 40]); right["target_step"] = right["target_step"].astype(float); right.loc[1, "target_step"] = 1.5
    left.to_csv(tmp_path / "lstm_predictions.csv", index=False)
    right.to_csv(tmp_path / "gru_predictions.csv", index=False)
    with pytest.raises(ValueError, match="differ"):
        load_prediction_files(tmp_path / "lstm_predictions.csv", tmp_path / "gru_predictions.csv")


def test_replay_two_models_and_summary():
    lstm = normalize_predictions(_raw("B", [150, 110, 90, 80], [60, 40, 20, 0]), "LSTM")
    gru = normalize_predictions(_raw("B", [150, 130, 125, 122], [60, 40, 20, 0]), "GRU")
    events, summary, timeline = run_replay(pd.concat([lstm, gru]), ReplaySettings(120, 2, 10))
    assert len(events) == 2
    assert len(timeline) == 8
    assert dict(zip(events.model, events.status)) == {"GRU": "missed", "LSTM": "timely"}
    assert dict(zip(summary.model, summary.timely)) == {"GRU": 0, "LSTM": 1}


def test_cli_writes_real_output_files_using_synthetic_fixture(tmp_path):
    evaluation = tmp_path / "evaluation"; evaluation.mkdir()
    _raw("B", [150, 110, 90, 80], [60, 40, 20, 0]).to_csv(evaluation / "lstm_predictions.csv", index=False)
    _raw("B", [150, 130, 125, 122], [60, 40, 20, 0]).to_csv(evaluation / "gru_predictions.csv", index=False)
    destination = tmp_path / "out"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_maintenance_replay.py"),
        "--evaluation-dir", str(evaluation), "--output-dir", str(destination),
        "--consecutive-snapshots", "2", "--minimum-lead-minutes", "10",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    for filename in ("replay_events.csv", "replay_summary.csv", "replay_timeline.csv", "replay_config.json"):
        assert (destination / filename).is_file()
