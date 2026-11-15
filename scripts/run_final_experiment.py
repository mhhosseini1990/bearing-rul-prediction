"""Run or verify the approved 30-checkpoint fixed48 direct-RUL experiment.

Normal execution trains 5 folds x 2 architectures x 3 seeds.  Completed
checkpoints are reused only after their complete provenance has been verified.
Use ``--check-only`` for a read-only readiness check and ``--summary-only`` to
validate and display a completed experiment without training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train as core  # noqa: E402

SEQUENCE_BY_FOLD = {"fold_1": 10, "fold_2": 30, "fold_3": 30, "fold_4": 10, "fold_5": 10}
COMMON_MIN_STEP = 29
OUTPUT_ACTIVATION = "softplus"
HIDDEN_SIZE = 32
BATCH_SIZE = 128
MAX_EPOCHS = 40
MIN_EPOCHS = 8
PATIENCE = 7
EXPECTED_CHECKPOINTS = 30


def tasks():
    return [(fold, model, seed) for fold in core.FOLDS for model in core.MODELS for seed in core.SEEDS]


def expected_metadata(fold, model, seed, data_hash):
    split = core.FOLDS[fold]
    return {
        "protocol": "fixed48_v1_direct_rul_final",
        "fold": fold, "model": model, "seed": seed,
        "sequence_length": SEQUENCE_BY_FOLD[fold],
        "common_test_min_step": COMMON_MIN_STEP,
        "input_size": 48, "hidden_size": HIDDEN_SIZE, "num_layers": 1,
        "output_activation": OUTPUT_ACTIVATION,
        "batch_size": BATCH_SIZE, "optimizer": "AdamW", "loss": "SmoothL1",
        "epochs_max": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE,
        "feature_set": core.FEATURE_SET_VERSION,
        "feature_registry_sha256": core.FEATURE_SET_SHA256,
        "csv_sha256": data_hash,
        "train_bearings": list(split["train"]), "val_bearings": list(split["val"]),
        "test_bearings": list(split["test"]),
    }


def verify_checkpoint(checkpoint, expected):
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        raise ValueError(f"Checkpoint provenance mismatch: {mismatches}")
    for key in ("model_state_dict", "feature_mean", "feature_std", "target_scale",
                "best_epoch", "validation_macro_mae_minutes", "learning_curve"):
        if key not in checkpoint:
            raise ValueError(f"Incomplete checkpoint: missing {key}")


def load_model(checkpoint):
    cls = core.MODELS[checkpoint["model"]]
    model = cls(input_size=48, hidden_size=HIDDEN_SIZE, num_layers=1,
                output_activation=OUTPUT_ACTIVATION)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def config(data_hash):
    return {
        "protocol": "fixed48_v1_direct_rul_final",
        "scientific_status": "exploratory; outer results were previously inspected",
        "feature_csv": "data/processed/xjtu_fixed_features_48.csv",
        "csv_sha256": data_hash,
        "feature_registry_sha256": core.FEATURE_SET_SHA256,
        "sequence_length_by_fold": SEQUENCE_BY_FOLD,
        "common_test_min_step": COMMON_MIN_STEP,
        "models": list(core.MODELS), "seeds": list(core.SEEDS),
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "input_size": 48, "hidden_size": HIDDEN_SIZE, "num_layers": 1,
        "output_activation": OUTPUT_ACTIVATION, "batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "loss": "SmoothL1",
        "epochs_max": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE,
        "sampling": "bearing-balanced WeightedRandomSampler on training only",
        "checkpoint_selection": "minimum validation bearing-macro MAE only",
        "folds": core.FOLDS,
    }


def aggregate(output, raw_predictions, metrics, curves, runs, data_hash):
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(raw_predictions, ignore_index=True)
    per_bearing = pd.concat(metrics, ignore_index=True)
    learning = pd.concat(curves, ignore_index=True)
    run_table = pd.DataFrame(runs)
    common = raw[raw.step >= COMMON_MIN_STEP].copy()
    if common.bearing_id.nunique() != 15:
        raise RuntimeError("Common test range does not cover all 15 bearings")
    raw.to_csv(output / "predictions_raw.csv", index=False)
    common.to_csv(output / "predictions_common_range.csv", index=False)
    per_bearing.to_csv(output / "per_bearing_metrics.csv", index=False)
    learning.to_csv(output / "learning_curves.csv", index=False)
    run_table.to_csv(output / "per_run_summary.csv", index=False)

    by_fold = per_bearing.groupby(["fold", "model", "seed"], as_index=False).agg(
        macro_mae_minutes=("mae_minutes", "mean"),
        macro_rmse_minutes=("rmse_minutes", "mean"), bearings=("bearing_id", "nunique"))
    by_condition = per_bearing.groupby(["condition", "model", "seed"], as_index=False).agg(
        macro_mae_minutes=("mae_minutes", "mean"),
        macro_rmse_minutes=("rmse_minutes", "mean"), bearings=("bearing_id", "nunique"))
    summary = per_bearing.groupby(["model", "seed"], as_index=False).agg(
        macro_mae_minutes=("mae_minutes", "mean"),
        macro_rmse_minutes=("rmse_minutes", "mean"), bearings=("bearing_id", "nunique"))
    model_summary = summary.groupby("model", as_index=False).agg(
        mean_macro_mae_minutes=("macro_mae_minutes", "mean"),
        std_macro_mae_minutes=("macro_mae_minutes", "std"),
        mean_macro_rmse_minutes=("macro_rmse_minutes", "mean"),
        std_macro_rmse_minutes=("macro_rmse_minutes", "std"))
    paired = per_bearing.pivot(index=["bearing_id", "condition", "seed"], columns="model",
                               values="mae_minutes").reset_index()
    paired["gru_minus_lstm_mae_minutes"] = paired["gru"] - paired["lstm"]
    by_fold.to_csv(output / "per_fold_metrics.csv", index=False)
    by_condition.to_csv(output / "per_condition_metrics.csv", index=False)
    summary.to_csv(output / "summary_by_seed.csv", index=False)
    model_summary.to_csv(output / "model_summary.csv", index=False)
    paired.to_csv(output / "paired_bearing_comparison.csv", index=False)
    (output / "split_manifest.json").write_text(json.dumps(core.FOLDS, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(config(data_hash), indent=2), encoding="utf-8")
    (output / "LIMITATIONS.txt").write_text(
        "XJTU-SY contains only 15 laboratory bearing histories. Earlier research inspected "
        "some outer-test outcomes, so these results are exploratory rather than a fresh "
        "untouched confirmation. Macro error does not establish deployment safety or "
        "industrial maintenance savings.\n", encoding="utf-8")
    print(model_summary.to_string(index=False))
    return model_summary


def run(args):
    frame, data_hash = core.validate_inputs(args.features)
    planned = tasks()
    if len(planned) != EXPECTED_CHECKPOINTS:
        raise RuntimeError("Internal task count is not exactly 30")
    print(f"READY: {len(frame)} snapshots, 15 bearings, 48 fixed features")
    print(f"CSV SHA256: {data_hash}")
    print(f"Registry SHA256: {core.FEATURE_SET_SHA256}")
    print(f"Approved sequences: {SEQUENCE_BY_FOLD}; common test step >= {COMMON_MIN_STEP}")
    print("Planned: 5 folds x 2 models x 3 seeds = 30 checkpoints")
    if args.check_only:
        existing = sorted((args.output / "checkpoints").glob("*.pt")) if args.output.exists() else []
        for path in existing:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            key = (checkpoint.get("fold"), checkpoint.get("model"), checkpoint.get("seed"))
            if key not in planned:
                raise ValueError(f"Unexpected checkpoint: {path}")
            verify_checkpoint(checkpoint, expected_metadata(*key, data_hash))
        print(f"CHECK ONLY: {len(existing)} verified existing checkpoints; no training or writes")
        return
    if args.summary_only:
        path = args.output / "model_summary.csv"
        if not path.is_file():
            raise FileNotFoundError("Final model summary is unavailable; experiment is incomplete")
        checkpoints = list((args.output / "checkpoints").glob("*.pt"))
        if len(checkpoints) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(f"Incomplete experiment: {len(checkpoints)}/30 checkpoints")
        for path_ckpt in checkpoints:
            checkpoint = torch.load(path_ckpt, map_location="cpu", weights_only=False)
            verify_checkpoint(checkpoint, expected_metadata(
                checkpoint["fold"], checkpoint["model"], checkpoint["seed"], data_hash))
        print(pd.read_csv(path).to_string(index=False))
        return

    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_predictions, metrics, curves, runs = [], [], [], []
    for index, (fold, model_name, seed) in enumerate(planned, 1):
        split = core.FOLDS[fold]
        sequence_length = SEQUENCE_BY_FOLD[fold]
        expected = expected_metadata(fold, model_name, seed, data_hash)
        path = checkpoint_dir / f"{fold}__{model_name}__seed{seed}.pt"
        train = frame[frame.bearing_id.isin(split["train"])].copy()
        val = frame[frame.bearing_id.isin(split["val"])].copy()
        test = frame[frame.bearing_id.isin(split["test"])].copy()
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            verify_checkpoint(checkpoint, expected)
            model = load_model(checkpoint)
            mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
            scale = float(checkpoint["target_scale"])
            best_epoch = int(checkpoint["best_epoch"])
            val_mae = float(checkpoint["validation_macro_mae_minutes"])
            curve = pd.DataFrame(checkpoint["learning_curve"])
            status = "resumed"
        else:
            values = train.loc[:, core.FEATURE_NAMES].to_numpy(dtype=np.float64)
            mean = values.mean(axis=0).astype(np.float32)
            std = np.where(values.std(axis=0) < 1e-8, 1.0, values.std(axis=0)).astype(np.float32)
            x_train, y_train, b_train, _ = core.sequences(train, sequence_length, mean, std)
            x_val, y_val, _, m_val = core.sequences(val, sequence_length, mean, std)
            model, scale, best_epoch, val_mae, curve = core.fit_one(
                x_train, y_train, b_train, x_val, y_val, m_val, model_name,
                HIDDEN_SIZE, seed, MAX_EPOCHS, PATIENCE, MIN_EPOCHS, BATCH_SIZE,
                output_activation=OUTPUT_ACTIVATION)
            checkpoint = {**expected, "model_state_dict": model.state_dict(),
                          "feature_names": list(core.FEATURE_NAMES),
                          "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                          "target_scale": scale, "best_epoch": best_epoch,
                          "validation_macro_mae_minutes": val_mae,
                          "learning_curve": curve.to_dict(orient="records")}
            temp = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temp)
            temp.replace(path)
            status = "trained"
        x_test, y_test, _, meta = core.sequences(test, sequence_length, mean, std)
        prediction = core.infer(model, x_test, scale, BATCH_SIZE)
        raw = meta.copy()
        raw["predicted_rul_minutes"] = prediction
        raw.insert(0, "seed", seed); raw.insert(0, "model", model_name); raw.insert(0, "fold", fold)
        common_mask = meta.step.to_numpy() >= COMMON_MIN_STEP
        per_bearing = core.score(y_test[common_mask], prediction[common_mask],
                                 meta.loc[common_mask].reset_index(drop=True))
        per_bearing.insert(0, "seed", seed); per_bearing.insert(0, "model", model_name)
        per_bearing.insert(0, "fold", fold)
        curve.insert(0, "seed", seed); curve.insert(0, "model", model_name); curve.insert(0, "fold", fold)
        raw_predictions.append(raw); metrics.append(per_bearing); curves.append(curve)
        runs.append({"fold": fold, "model": model_name, "seed": seed,
                     "sequence_length": sequence_length, "best_epoch": best_epoch,
                     "validation_macro_mae_minutes": val_mae,
                     "test_common_macro_mae_minutes": float(per_bearing.mae_minutes.mean()),
                     "test_common_macro_rmse_minutes": float(per_bearing.rmse_minutes.mean())})
        print(f"[{index}/30] {status}: {fold} {model_name} seed={seed} best_epoch={best_epoch}", flush=True)
    aggregate(args.output, raw_predictions, metrics, curves, runs, data_hash)
    checkpoint_paths = list(checkpoint_dir.glob("*.pt"))
    if len(checkpoint_paths) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Incomplete experiment: {len(checkpoint_paths)}/30 checkpoints")
    print("COMPLETE: exactly 30 provenance-verified checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/xjtu_fixed_features_48.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/model/fixed48_direct_final")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--summary-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
