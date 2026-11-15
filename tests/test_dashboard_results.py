"""Result discovery and checkpoint-completion tests for the dashboard."""
import json
from pathlib import Path

import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from dashboard.results import discover_results

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/final/model/fixed48_direct_final"
SEQUENCES = {"fold_1": 10, "fold_2": 30, "fold_3": 30, "fold_4": 10, "fold_5": 10}


def _config():
    return {"sequence_length_by_fold": SEQUENCES, "csv_sha256": "csv-hash",
            "feature_registry_sha256": "registry-hash"}


def _checkpoint(fold="fold_1", model="gru", seed=42, csv_hash="csv-hash"):
    return {"protocol": "fixed48_v1_direct_rul_final", "fold": fold, "model": model,
            "seed": seed, "sequence_length": SEQUENCES[fold], "common_test_min_step": 29,
            "input_size": 48, "hidden_size": 32, "num_layers": 1,
            "output_activation": "softplus", "batch_size": 128, "optimizer": "AdamW",
            "loss": "SmoothL1", "epochs_max": 40, "min_epochs": 8, "patience": 7,
            "feature_set": "fixed48_v1", "feature_registry_sha256": "registry-hash",
            "csv_sha256": csv_hash, "train_bearings": list(range(9)),
            "val_bearings": list(range(3)), "test_bearings": list(range(3)),
            "model_state_dict": {}, "feature_mean": [0.] * 48, "feature_std": [1.] * 48,
            "target_scale": 1., "best_epoch": 8, "validation_macro_mae_minutes": 1.,
            "learning_curve": []}


def _write_minimum_tables(root, run):
    pd.DataFrame([{"model": "gru", "mean_macro_mae_minutes": 1.,
                   "std_macro_mae_minutes": 0., "mean_macro_rmse_minutes": 2.,
                   "std_macro_rmse_minutes": 0.}]).to_csv(root / "model_summary.csv", index=False)
    pd.DataFrame([run]).to_csv(root / "per_run_summary.csv", index=False)


def test_actual_final_experiment_has_thirty_verified_runs():
    state = discover_results(FINAL)
    assert state["complete"]
    assert state["checkpoint_files"] == state["verified_count"] == state["completed_count"] == 30
    assert state["missing"] == [] and state["invalid"] == [] and state["duplicates"] == []
    assert set(state["model_summary"].model) == {"lstm", "gru"}


def test_partial_experiment_reports_missing_runs_without_hiding_available_result(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "run_config.json").write_text(json.dumps(_config()), encoding="utf-8")
    torch.save(_checkpoint(), tmp_path / "checkpoints/fold_1__gru__seed42.pt")
    run = {"fold": "fold_1", "model": "gru", "seed": 42, "sequence_length": 10,
           "best_epoch": 8, "validation_macro_mae_minutes": 1.,
           "test_common_macro_mae_minutes": 2., "test_common_macro_rmse_minutes": 3.}
    _write_minimum_tables(tmp_path, run)
    state = discover_results(tmp_path)
    assert not state["complete"]
    assert state["verified_count"] == state["completed_count"] == 1
    assert len(state["missing"]) == 29
    assert not state["model_summary"].empty


def test_invalid_checkpoint_provenance_is_reported(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "run_config.json").write_text(json.dumps(_config()), encoding="utf-8")
    torch.save(_checkpoint(csv_hash="wrong"), tmp_path / "checkpoints/fold_1__gru__seed42.pt")
    run = {"fold": "fold_1", "model": "gru", "seed": 42, "sequence_length": 10,
           "best_epoch": 8, "validation_macro_mae_minutes": 1.,
           "test_common_macro_mae_minutes": 2., "test_common_macro_rmse_minutes": 3.}
    _write_minimum_tables(tmp_path, run)
    state = discover_results(tmp_path)
    assert state["verified_count"] == state["completed_count"] == 0
    assert state["invalid"] and "provenance mismatch" in state["invalid"][0]["reason"]


def test_dashboard_overview_displays_verified_completion():
    app = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=60).run()
    assert not app.exception
    assert any("all 30" in item.value.lower() for item in app.success)
    metric_values = {item.label: item.value for item in app.metric}
    assert metric_values["Verified checkpoints"] == "30/30"
    assert metric_values["Completed runs"] == "30/30"
