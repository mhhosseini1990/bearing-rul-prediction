"""Discover and verify fixed48 final-experiment artifacts for the dashboard."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch

FOLDS = tuple(f"fold_{index}" for index in range(1, 6))
MODELS = ("lstm", "gru")
SEEDS = (42, 123, 2026)
EXPECTED_RUNS = {(fold, model, seed) for fold in FOLDS for model in MODELS for seed in SEEDS}
CHECKPOINT_PATTERN = re.compile(r"^(fold_[1-5])__(lstm|gru)__seed(42|123|2026)\.pt$")
MODEL_SUMMARY_COLUMNS = {
    "model", "mean_macro_mae_minutes", "std_macro_mae_minutes",
    "mean_macro_rmse_minutes", "std_macro_rmse_minutes",
}
RUN_COLUMNS = {"fold", "model", "seed", "sequence_length", "best_epoch",
               "validation_macro_mae_minutes", "test_common_macro_mae_minutes",
               "test_common_macro_rmse_minutes"}
REQUIRED_CHECKPOINT_KEYS = {
    "protocol", "fold", "model", "seed", "sequence_length", "common_test_min_step",
    "input_size", "hidden_size", "num_layers", "output_activation", "batch_size",
    "optimizer", "loss", "epochs_max", "min_epochs", "patience", "feature_set",
    "feature_registry_sha256", "csv_sha256", "train_bearings", "val_bearings",
    "test_bearings", "model_state_dict", "feature_mean", "feature_std", "target_scale",
    "best_epoch", "validation_macro_mae_minutes", "learning_curve",
}


def _csv(path: Path, required: set[str]) -> tuple[pd.DataFrame, str | None]:
    if not path.is_file():
        return pd.DataFrame(), f"missing {path.name}"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), f"cannot read {path.name}: {exc}"
    missing = required - set(frame.columns)
    return (frame, None) if not missing else (pd.DataFrame(), f"{path.name} missing columns: {sorted(missing)}")


def _checkpoint_error(path: Path, config: dict) -> tuple[tuple[str, str, int] | None, str | None]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if not match:
        return None, "filename does not match the final protocol"
    filename_key = (match.group(1), match.group(2), int(match.group(3)))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"cannot load: {exc}"
    missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        return None, f"missing metadata: {sorted(missing)}"
    metadata_key = (checkpoint["fold"], checkpoint["model"], int(checkpoint["seed"]))
    if metadata_key != filename_key:
        return None, f"filename/metadata mismatch: {metadata_key}"
    sequence_by_fold = config.get("sequence_length_by_fold", {})
    expected = {
        "protocol": "fixed48_v1_direct_rul_final", "feature_set": "fixed48_v1",
        "input_size": 48, "hidden_size": 32, "num_layers": 1,
        "output_activation": "softplus", "batch_size": 128,
        "optimizer": "AdamW", "loss": "SmoothL1", "epochs_max": 40,
        "min_epochs": 8, "patience": 7, "common_test_min_step": 29,
        "sequence_length": sequence_by_fold.get(filename_key[0]),
        "csv_sha256": config.get("csv_sha256"),
        "feature_registry_sha256": config.get("feature_registry_sha256"),
    }
    mismatches = [name for name, value in expected.items() if value is None or checkpoint.get(name) != value]
    if mismatches:
        return None, f"provenance mismatch: {mismatches}"
    if (len(checkpoint["train_bearings"]), len(checkpoint["val_bearings"]),
            len(checkpoint["test_bearings"])) != (9, 3, 3):
        return None, "checkpoint split is not 9/3/3"
    return filename_key, None


def discover_results(root: Path) -> dict:
    """Return verified completion state and all displayable result tables."""
    root = Path(root)
    errors: list[str] = []
    config_path = root / "run_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            config = {}; errors.append(f"cannot read run_config.json: {exc}")
    else:
        config = {}; errors.append("missing run_config.json")

    summary, error = _csv(root / "model_summary.csv", MODEL_SUMMARY_COLUMNS)
    if error: errors.append(error)
    runs, error = _csv(root / "per_run_summary.csv", RUN_COLUMNS)
    if error: errors.append(error)
    fold_metrics, error = _csv(root / "per_fold_metrics.csv",
                               {"fold", "model", "seed", "macro_mae_minutes", "macro_rmse_minutes"})
    if error: errors.append(error)
    seed_metrics, error = _csv(root / "summary_by_seed.csv",
                               {"model", "seed", "macro_mae_minutes", "macro_rmse_minutes"})
    if error: errors.append(error)
    curves, error = _csv(root / "learning_curves.csv",
                         {"fold", "model", "seed", "epoch", "train_loss", "val_macro_mae_minutes"})
    if error: errors.append(error)

    verified: list[tuple[str, str, int]] = []
    invalid: list[dict] = []
    checkpoint_dir = root / "checkpoints"
    checkpoint_files = sorted(
        path for suffix in ("*.pt", "*.pth", "*.ckpt")
        for path in checkpoint_dir.glob(suffix)
    ) if checkpoint_dir.is_dir() else []
    for path in checkpoint_files:
        key, checkpoint_error = _checkpoint_error(path, config)
        if checkpoint_error:
            invalid.append({"file": path.name, "reason": checkpoint_error})
        else:
            verified.append(key)
    duplicates = sorted({key for key in verified if verified.count(key) > 1})
    verified_set = set(verified)
    run_keys = set()
    if not runs.empty:
        for row in runs[["fold", "model", "seed"]].itertuples(index=False, name=None):
            try:
                run_keys.add((str(row[0]), str(row[1]), int(row[2])))
            except (TypeError, ValueError):
                errors.append(f"invalid run key: {row}")
    completed = verified_set & run_keys & EXPECTED_RUNS
    missing = sorted(EXPECTED_RUNS - completed)
    complete = (len(completed) == 30 and not missing and not duplicates and not invalid
                and len(run_keys) == 30 and not summary.empty)
    return {
        "complete": complete, "expected_count": 30, "checkpoint_files": len(checkpoint_files),
        "verified_count": len(verified_set), "completed_count": len(completed),
        "missing": missing, "duplicates": duplicates, "invalid": invalid, "errors": errors,
        "model_summary": summary, "runs": runs, "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics, "learning_curves": curves, "config": config,
    }
