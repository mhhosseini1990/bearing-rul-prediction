"""Train direct-RUL LSTM/GRU models using the fixed 48-feature XJTU-SY matrix.

Protocol: five bearing-disjoint outer folds, exactly 9 training / 3 validation /
3 test bearings per fold (3/1/1 per operating condition); three seeds per
model; validation-only early stopping; no health indicator or feature selection.

First run (no training, no writes):
    python scripts/train.py --check-only

Approved full training is orchestrated only by:
    python scripts/run_final_experiment.py

Run only one model for a smoke test:
    python scripts/train.py --fold fold_1 --model gru --seed 42 --epochs 8

Existing experiment outputs are not overwritten without --overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_rul.features.fixed_features import (  # noqa: E402
    FEATURE_NAMES, FEATURE_SET_SHA256, FEATURE_SET_VERSION,
)
from bearing_rul.models.gru import GRURegressor  # noqa: E402
from bearing_rul.models.lstm import LSTMRegressor  # noqa: E402

MODELS = {"lstm": LSTMRegressor, "gru": GRURegressor}
SEEDS = (42, 123, 2026)
CONDITIONS = (1, 2, 3)
FEATURE_COUNT = 48
META_COLUMNS = ("bearing_id", "condition", "step", "elapsed_minutes", "rul_minutes")


def build_folds() -> dict[str, dict[str, tuple[str, ...]]]:
    """Predeclared 9/3/3 bearing splits; never changed based on test performance.

    Fold 1 uses Bearing3_4 for validation and Bearing3_2 for training,
    rather than the standard validation rotation. This *pre-training* change
    improves the range of Condition-3 training lifetimes in Fold 1. All outer
    test bearings and the other four folds retain their original allocation.
    """
    folds = {}
    for i in range(5):
        test_number = i + 1
        default_val_number = ((i + 1) % 5) + 1
        val_by_condition = {condition: default_val_number for condition in CONDITIONS}
        if i == 0:
            val_by_condition[3] = 4  # Fold 1 only: B3_4 val; B3_2 train
        folds[f"fold_{i+1}"] = {
            "train": tuple(
                f"Bearing{condition}_{j}"
                for condition in CONDITIONS
                for j in range(1, 6)
                if j not in (test_number, val_by_condition[condition])
            ),
            "val": tuple(f"Bearing{condition}_{val_by_condition[condition]}" for condition in CONDITIONS),
            "test": tuple(f"Bearing{condition}_{test_number}" for condition in CONDITIONS),
        }
    return folds


FOLDS = build_folds()


def sha256_file(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(part)
    return hash_.hexdigest()


def validate_inputs(csv_path: Path) -> tuple[pd.DataFrame, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing feature CSV: {csv_path}. Run scripts/prepare_data.py --extract first.")
    manifest_path = csv_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(csv_path)
    expected = {
        "feature_set": FEATURE_SET_VERSION,
        "feature_registry_sha256": FEATURE_SET_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "sha256": digest,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Feature manifest mismatch: {mismatches}; rerun preparation/validation")

    frame = pd.read_csv(csv_path)
    required_columns = (*META_COLUMNS, *FEATURE_NAMES)
    if tuple(frame.columns) != required_columns:
        raise ValueError("CSV column names/order differ from fixed48_v1 feature registry")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Empty matrix or missing values")
    numbers = frame.loc[:, ["step", "elapsed_minutes", "rul_minutes", *FEATURE_NAMES]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Non-finite input feature, time, or RUL values")
    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Duplicate bearing_id/step rows")
    expected_ids = {f"Bearing{c}_{j}" for c in CONDITIONS for j in range(1, 6)}
    if set(frame.bearing_id.unique()) != expected_ids:
        raise ValueError("Expected exactly 15 XJTU-SY bearings (5 per condition)")
    for bearing, part in frame.groupby("bearing_id"):
        steps = np.sort(part.step.to_numpy())
        if not np.array_equal(steps, np.arange(len(steps))):
            raise ValueError(f"Invalid consecutive steps in {bearing}")
        if part.condition.nunique() != 1:
            raise ValueError(f"Multiple conditions found for {bearing}")
        if not np.array_equal(part.rul_minutes.to_numpy(), len(part) - 1 - part.step.to_numpy()):
            raise ValueError(f"Invalid RUL definition in {bearing}")
    validate_folds(expected_ids)
    return frame, digest


def validate_folds(all_ids: set[str]) -> None:
    test_occurrences = []
    for fold, split in FOLDS.items():
        a, b, c = map(set, (split["train"], split["val"], split["test"]))
        if (len(a), len(b), len(c)) != (9, 3, 3) or a & b or a & c or b & c:
            raise ValueError(f"Invalid 9/3/3 disjoint split: {fold}")
        if a | b | c != all_ids:
            raise ValueError(f"Missing or extra bearing in {fold}")
        for label, ids, expected_n in (("train", a, 3), ("val", b, 1), ("test", c, 1)):
            for cond in CONDITIONS:
                if sum(x.startswith(f"Bearing{cond}_") for x in ids) != expected_n:
                    raise ValueError(f"Operating condition imbalance in {fold}, {label}")
        test_occurrences.extend(split["test"])
    if sorted(test_occurrences) != sorted(all_ids):
        raise ValueError("Each bearing must be outer test exactly once")


def split_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive lifetime audit only; NEVER used to select a split."""
    lengths = frame.groupby("bearing_id").size().to_dict()
    records = []
    for fold, split in FOLDS.items():
        for part, ids in split.items():
            for bearing in ids:
                records.append({
                    "fold": fold, "split": part, "bearing_id": bearing,
                    "condition": int(bearing.split("_")[0].replace("Bearing", "")),
                    "lifetime_minutes": lengths[bearing] - 1,
                })
    return pd.DataFrame(records)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sequences(frame: pd.DataFrame, sequence_length: int, mean: np.ndarray,
              std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    xs, targets, owners, metadata = [], [], [], []
    for bearing, rows in frame.groupby("bearing_id", sort=True):
        rows = rows.sort_values("step")
        if len(rows) < sequence_length:
            raise ValueError(f"{bearing} has fewer than {sequence_length} snapshots")
        signal = (rows.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32) - mean) / std
        rul = rows.rul_minutes.to_numpy(dtype=np.float32)
        for t in range(sequence_length - 1, len(rows)):
            xs.append(signal[t-sequence_length+1:t+1])
            targets.append(rul[t])
            owners.append(bearing)
            metadata.append((bearing, rows.condition.iloc[0], int(rows.step.iloc[t]), float(rul[t])))
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(owners),
        pd.DataFrame(metadata, columns=["bearing_id", "condition", "step", "rul_minutes"]),
    )


def infer(model: nn.Module, x: np.ndarray, target_scale: float, batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (xb,) in loader:
            raw = model(xb).reshape(-1).cpu().numpy()
            predictions.append(np.maximum(raw * target_scale, 0.0))
    return np.concatenate(predictions)


def score(truth: np.ndarray, prediction: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()
    result["predicted_rul_minutes"] = prediction
    records = []
    for bearing, rows in result.groupby("bearing_id", sort=True):
        error = rows.predicted_rul_minutes.to_numpy() - rows.rul_minutes.to_numpy()
        records.append({
            "bearing_id": bearing,
            "condition": rows.condition.iloc[0],
            "mae_minutes": float(np.abs(error).mean()),
            "rmse_minutes": float(np.sqrt(np.square(error).mean())),
            "evaluated_snapshots": len(rows),
        })
    return pd.DataFrame(records)


def fit_one(x_train: np.ndarray, y_train: np.ndarray, b_train: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, m_val: pd.DataFrame,
            model_name: str, hidden: int, seed: int,
            epochs: int, patience: int, min_epochs: int, batch_size: int,
            output_activation: str = "linear"):
    set_seed(seed)
    # A common fixed architecture (not selected based on outer test performance).
    model = MODELS[model_name](input_size=FEATURE_COUNT, hidden_size=hidden, num_layers=1,
                               output_activation=output_activation)
    target_scale = max(float(np.quantile(y_train, 0.95)), 1.0)
    counts = pd.Series(b_train).value_counts()
    weights = pd.Series(b_train).map(lambda key: 1.0 / counts[key]).to_numpy(dtype=float)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy((y_train / target_scale).reshape(-1, 1))),
        batch_size=batch_size, sampler=sampler,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = math.inf
    best_epoch = 0
    best_weights = None
    stale = 0
    learning_curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(xb)
            seen += len(xb)
        val_predictions = infer(model, x_val, target_scale, batch_size)
        val_per_bearing = score(y_val, val_predictions, m_val)
        val_macro_mae = float(val_per_bearing.mae_minutes.mean())
        learning_curve.append({"epoch": epoch, "train_loss": running_loss / seen,
                               "val_macro_mae_minutes": val_macro_mae})
        if val_macro_mae < best_score - 1e-6:
            best_score = val_macro_mae
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    if best_weights is None:
        raise RuntimeError("No valid validation checkpoint")
    model.load_state_dict(best_weights)
    model.eval()
    return model, target_scale, best_epoch, best_score, pd.DataFrame(learning_curve)


def run(args: argparse.Namespace) -> None:
    if args.check_only and args.validation_only:
        raise ValueError("Use either --check-only or --validation-only, not both")
    if args.sequence < 2 or args.batch_size < 1 or args.hidden_size < 1:
        raise ValueError("Invalid sequence, batch size or hidden size")
    if args.epochs < 1 or args.patience < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        raise ValueError("Require 1 <= min-epochs <= epochs and patience >= 1")
    frame, data_hash = validate_inputs(args.features)
    audit = split_audit(frame)
    print(f"DATA VALID: {len(frame)} snapshots, 15 bearings, {FEATURE_COUNT} fixed features")
    print(f"Feature set: {FEATURE_SET_VERSION} | registry SHA256: {FEATURE_SET_SHA256}")
    for fold, split in FOLDS.items():
        print(f"{fold}: TRAIN={','.join(split['train'])} | VAL={','.join(split['val'])} | TEST={','.join(split['test'])}")
    if not args.validation_only:
        print("Lifetime balance audit (minutes; not used for choosing folds):")
        print(audit.groupby(["fold", "split"]).lifetime_minutes.agg(["min", "median", "max"]).to_string())
    if args.check_only:
        print("CHECK ONLY: no training and no files written")
        return

    tasks = []
    for fold in FOLDS if args.fold == "all" else [args.fold]:
        for model_name in MODELS if args.model == "all" else [args.model]:
            for seed in SEEDS if args.seed == "all" else [int(args.seed)]:
                tasks.append((fold, model_name, seed))
    checkpoints = args.output / "checkpoints"
    if not args.overwrite:
        existing = [checkpoints / f"{fold}__{model}__seed{seed}.pt"
                    for fold, model, seed in tasks if (checkpoints / f"{fold}__{model}__seed{seed}.pt").exists()]
        if existing:
            raise FileExistsError(f"Will not overwrite checkpoint: {existing[0]}. Use --overwrite explicitly.")
    checkpoints.mkdir(parents=True, exist_ok=True)

    all_predictions, all_metrics, all_curves, all_runs = [], [], [], []
    for fold, model_name, seed in tasks:
        split = FOLDS[fold]
        train = frame[frame.bearing_id.isin(split["train"])].copy()
        val = frame[frame.bearing_id.isin(split["val"])].copy()
        if not args.validation_only:
            test = frame[frame.bearing_id.isin(split["test"])].copy()
        # ONLY 9 train bearings are used for feature normalization and target scaling.
        train_values = train.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
        feature_mean = train_values.mean(axis=0).astype(np.float32)
        feature_std = train_values.std(axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)
        x_train, y_train, b_train, _ = sequences(train, args.sequence, feature_mean, feature_std)
        x_val, y_val, _, m_val = sequences(val, args.sequence, feature_mean, feature_std)
        if not args.validation_only:
            x_test, y_test, _, m_test = sequences(test, args.sequence, feature_mean, feature_std)
        model, scale, best_epoch, val_mae, curve = fit_one(
            x_train, y_train, b_train, x_val, y_val, m_val,
            model_name, args.hidden_size, seed, args.epochs, args.patience,
            args.min_epochs, args.batch_size, args.output_activation,
        )
        # Strict validation-only mode: do not form test sequences, predictions,
        # or test metrics. Hyperparameter comparisons must use validation data.
        if args.validation_only:
            evaluated_pred = infer(model, x_val, scale, args.batch_size)
            evaluated_metric = score(y_val, evaluated_pred, m_val)
            evaluated_metadata = m_val.copy()
        else:
            evaluated_pred = infer(model, x_test, scale, args.batch_size)
            evaluated_metric = score(y_test, evaluated_pred, m_test)
            evaluated_metadata = m_test.copy()
        evaluated_metric.insert(0, "seed", seed)
        evaluated_metric.insert(0, "model", model_name)
        evaluated_metric.insert(0, "fold", fold)
        evaluated_metadata["predicted_rul_minutes"] = evaluated_pred
        evaluated_metadata.insert(0, "seed", seed)
        evaluated_metadata.insert(0, "model", model_name)
        evaluated_metadata.insert(0, "fold", fold)
        curve.insert(0, "seed", seed)
        curve.insert(0, "model", model_name)
        curve.insert(0, "fold", fold)
        all_predictions.append(evaluated_metadata)
        all_metrics.append(evaluated_metric)
        all_curves.append(curve)
        run_result = {
            "fold": fold, "model": model_name, "seed": seed,
            "output_activation": args.output_activation,
            "best_epoch": best_epoch,
            "validation_macro_mae_minutes": val_mae,
        }
        if args.validation_only:
            run_result["validation_macro_rmse_minutes"] = float(evaluated_metric.rmse_minutes.mean())
        else:
            run_result["test_macro_mae_minutes"] = float(evaluated_metric.mae_minutes.mean())
            run_result["test_macro_rmse_minutes"] = float(evaluated_metric.rmse_minutes.mean())
        all_runs.append(run_result)
        checkpoint = {
            "model_state_dict": model.state_dict(), "model": model_name,
            "fold": fold, "seed": seed, "hidden_size": args.hidden_size,
            "num_layers": 1, "input_size": FEATURE_COUNT, "sequence_length": args.sequence,
            "output_activation": args.output_activation, "validation_only": args.validation_only,
            "feature_set": FEATURE_SET_VERSION, "feature_registry_sha256": FEATURE_SET_SHA256,
            "feature_names": list(FEATURE_NAMES), "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(), "target_scale": scale,
            "csv_sha256": data_hash, "train_bearings": split["train"],
            "val_bearings": split["val"], "test_bearings": split["test"],
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, checkpoints / f"{fold}__{model_name}__seed{seed}.pt")
        print(f"{fold} {model_name} seed={seed} activation={args.output_activation} "
              f"best_epoch={best_epoch} VAL_macro_MAE={val_mae:.2f}" +
              ("" if args.validation_only else
               f" TEST_macro_MAE={evaluated_metric.mae_minutes.mean():.2f}"), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.validation_only:
        audit.to_csv(args.output / "split_audit.csv", index=False)
    label = "validation" if args.validation_only else "test"
    pd.concat(all_predictions, ignore_index=True).to_csv(
        args.output / ("validation_predictions.csv" if args.validation_only else "predictions.csv"), index=False)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.output / ("validation_per_bearing_metrics.csv" if args.validation_only
                                  else "per_bearing_metrics.csv"), index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(args.output / "learning_curves.csv", index=False)
    pd.DataFrame(all_runs).to_csv(args.output / "per_run_summary.csv", index=False)
    summary = metrics.groupby(["model", "seed"], as_index=False).agg(
        macro_mae_minutes=("mae_minutes", "mean"),
        macro_rmse_minutes=("rmse_minutes", "mean"),
        evaluated_bearings=("bearing_id", "nunique"),
    )
    summary.insert(2, "evaluation_split", label)
    summary.insert(3, "output_activation", args.output_activation)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "run_config.json").write_text(json.dumps({
        "protocol": "fixed48_direct_rul_9train_3val_3test_5outer_folds",
        "validation_only": args.validation_only,
        "output_activation": args.output_activation,
        "features": str(args.features), "csv_sha256": data_hash,
        "feature_registry_sha256": FEATURE_SET_SHA256,
        "sequence_length": args.sequence, "hidden_size": args.hidden_size,
        "epochs_max": args.epochs, "patience": args.patience,
        "min_epochs": args.min_epochs,
        "seed_values": [seed for _, _, seed in tasks],
        "folds": FOLDS,
        "tasks_completed": len(tasks),
    }, indent=2), encoding="utf-8")
    print("\n" + ("VALIDATION-ONLY SUMMARY" if args.validation_only else "FINAL TEST SUMMARY")
          + " (lower MAE / RMSE is better):")
    print(summary.to_string(index=False))
    print(f"Saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=ROOT / "data/processed/xjtu_fixed_features_48.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/manual_training_run")
    parser.add_argument("--check-only", action="store_true", help="Audit 9/3/3 splits, no training or writing")
    parser.add_argument("--validation-only", action="store_true",
                        help="Train/select by validation; do not evaluate or write test predictions/metrics")
    parser.add_argument("--output-activation", choices=("softplus",), default="softplus",
                        help="Final protocol uses nonnegative Softplus")
    parser.add_argument("--fold", choices=("all", *FOLDS), default="all")
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--seed", choices=("all", *map(str, SEEDS)), default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
