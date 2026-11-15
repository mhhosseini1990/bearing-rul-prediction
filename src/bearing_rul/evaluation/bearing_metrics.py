
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def compute_mae_rmse_minutes(
    y_true_minutes: np.ndarray,
    y_pred_minutes: np.ndarray,
) -> Dict[str, float]:
    """Computes MAE and RMSE in minutes for paired actual and predicted RUL arrays."""
    yt = np.asarray(y_true_minutes, dtype=np.float64)
    yp = np.asarray(y_pred_minutes, dtype=np.float64)

    if yt.size == 0 or yp.size == 0:
        raise ValueError("Cannot compute RUL metrics on empty prediction arrays.")
    if yt.shape != yp.shape:
        raise ValueError(f"Array shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}.")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("NaN values detected in RUL prediction arrays.")

    errors = yp - yt
    return {
        "mae_minutes": float(np.mean(np.abs(errors))),
        "rmse_minutes": float(np.sqrt(np.mean(errors**2))),
    }


def evaluate_per_bearing_minutes(
    df: pd.DataFrame,
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    pred_col: str = "predicted_rul",
    minutes_per_step: float = 1.0,
) -> pd.DataFrame:
    """Computes per-bearing MAE, RMSE (in minutes), window counts, and aggregation weights.

    Exposes `pooled_window_weight` vs `equal_bearing_weight` so weight disparities caused
    by unequal bearing lifespans are directly inspectable.
    """
    required = {bearing_col, actual_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction DataFrame is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Prediction DataFrame is empty.")
    if minutes_per_step <= 0:
        raise ValueError("minutes_per_step must be strictly positive.")

    total_windows = int(len(df))
    grouped = list(df.groupby(bearing_col, sort=True))
    num_bearings = len(grouped)
    equal_weight = 1.0 / num_bearings

    rows = []
    for bearing_id, group in grouped:
        window_count = int(len(group))
        pooled_weight = window_count / total_windows
        y_true_min = group[actual_col].to_numpy(dtype=np.float64) * minutes_per_step
        y_pred_min = group[pred_col].to_numpy(dtype=np.float64) * minutes_per_step

        metrics = compute_mae_rmse_minutes(y_true_min, y_pred_min)
        rows.append(
            {
                bearing_col: bearing_id,
                "window_count": window_count,
                "pooled_window_weight": float(pooled_weight),
                "equal_bearing_weight": float(equal_weight),
                "weight_delta": float(equal_weight - pooled_weight),
                "mae_minutes": metrics["mae_minutes"],
                "rmse_minutes": metrics["rmse_minutes"],
            }
        )

    return pd.DataFrame(rows)


def build_inspectable_metric_report(
    df: pd.DataFrame,
    model_name: str = "model",
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    pred_col: str = "predicted_rul",
    minutes_per_step: float = 1.0,
) -> Dict[str, Any]:
    """Builds a full report comparing pooled window-weighted and equal-bearing mean metrics."""
    per_bearing_df = evaluate_per_bearing_minutes(
        df=df,
        bearing_col=bearing_col,
        actual_col=actual_col,
        pred_col=pred_col,
        minutes_per_step=minutes_per_step,
    )

    all_true_min = df[actual_col].to_numpy(dtype=np.float64) * minutes_per_step
    all_pred_min = df[pred_col].to_numpy(dtype=np.float64) * minutes_per_step
    pooled_raw = compute_mae_rmse_minutes(all_true_min, all_pred_min)

    pooled_window_weighted = {
        "aggregation_method": "pooled_window_weighted",
        "mae_minutes": pooled_raw["mae_minutes"],
        "rmse_minutes": pooled_raw["rmse_minutes"],
        "total_windows": int(len(df)),
        "n_bearings": int(len(per_bearing_df)),
    }

    equal_bearing_mean = {
        "aggregation_method": "equal_bearing_mean",
        "mae_minutes": float(per_bearing_df["mae_minutes"].mean()),
        "rmse_minutes": float(per_bearing_df["rmse_minutes"].mean()),
        "total_windows": int(len(df)),
        "n_bearings": int(len(per_bearing_df)),
    }

    aggregation_gap = {
        "mae_minutes_diff": float(
            equal_bearing_mean["mae_minutes"] - pooled_window_weighted["mae_minutes"]
        ),
        "rmse_minutes_diff": float(
            equal_bearing_mean["rmse_minutes"] - pooled_window_weighted["rmse_minutes"]
        ),
    }

    return {
        "model_name": model_name,
        "unit": "minutes",
        "minutes_per_step": float(minutes_per_step),
        "pooled_window_weighted": pooled_window_weighted,
        "equal_bearing_mean": equal_bearing_mean,
        "aggregation_gap": aggregation_gap,
        "per_bearing": per_bearing_df.to_dict(orient="records"),
    }


def verify_identical_prediction_targets(
    lstm_df: pd.DataFrame,
    gru_df: pd.DataFrame,
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    window_col: Optional[str] = None,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Verifies that LSTM and GRU predictions were evaluated on identical ground-truth targets."""
    required = {bearing_col, actual_col}
    if window_col:
        required.add(window_col)

    for label, frame in [("LSTM", lstm_df), ("GRU", gru_df)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions missing required columns: {sorted(missing)}")

    if len(lstm_df) != len(gru_df):
        raise ValueError(
            f"Total window count mismatch: LSTM has {len(lstm_df)} rows, GRU has {len(gru_df)} rows."
        )

    sort_cols = [bearing_col, window_col] if window_col else [bearing_col]
    lstm_sorted = lstm_df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    gru_sorted = gru_df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

    if not (lstm_sorted[bearing_col].to_numpy() == gru_sorted[bearing_col].to_numpy()).all():
        raise ValueError("Held-out bearing IDs or per-bearing window counts differ between LSTM and GRU.")

    if window_col and not (lstm_sorted[window_col].to_numpy() == gru_sorted[window_col].to_numpy()).all():
        raise ValueError(f"Window alignment column '{window_col}' differs between LSTM and GRU.")

    lstm_targets = lstm_sorted[actual_col].to_numpy(dtype=np.float64)
    gru_targets = gru_sorted[actual_col].to_numpy(dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(lstm_targets - gru_targets)))

    if not np.allclose(lstm_targets, gru_targets, rtol=0.0, atol=atol):
        raise ValueError(
            f"Ground-truth '{actual_col}' targets differ between LSTM and GRU "
            f"(max_abs_diff={max_abs_diff:.8f} > atol={atol})."
        )

    per_bearing_counts = (
        lstm_sorted.groupby(bearing_col, sort=True).size().to_dict()
    )
    return {
        "targets_identical": True,
        "total_windows": int(len(lstm_sorted)),
        "per_bearing_window_counts": {str(k): int(v) for k, v in per_bearing_counts.items()},
        "max_target_abs_diff": max_abs_diff,
    }


def recompute_metrics_from_saved_predictions(
    predictions_csv_path: Union[str, Path],
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    pred_col: str = "predicted_rul",
    minutes_per_step: float = 1.0,
) -> Dict[str, Any]:
    """Independently loads saved predictions from CSV, recomputes all metrics, and exports reports."""
    csv_path = Path(predictions_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Saved predictions file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    report = build_inspectable_metric_report(
        df=df,
        model_name=model_name,
        bearing_col=bearing_col,
        actual_col=actual_col,
        pred_col=pred_col,
        minutes_per_step=minutes_per_step,
    )
    report["source_predictions_path"] = str(csv_path)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report["per_bearing"]).to_csv(
            out_path / f"{model_name}_per_bearing_metrics.csv", index=False
        )
        with open(out_path / f"{model_name}_metric_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report