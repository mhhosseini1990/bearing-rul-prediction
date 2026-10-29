"""
Visualization for Rolling Bearing RUL Prediction.

Creates:
1. LSTM training history
2. GRU training history
3. MAE comparison
4. RMSE comparison
5. Per-bearing MAE comparison
6-11. Actual vs Predicted RUL for each model and test bearing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_chart(path: Path) -> None:
    """Save the current chart as a PNG file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path.name}", flush=True)


def plot_training_history(
    run_dir: Path,
    chart_dir: Path,
) -> None:
    """Plot training and validation losses."""

    for model in ("lstm", "gru"):

        history_path = (
            run_dir
            / "training"
            / f"{model}_history.csv"
        )

        if not history_path.is_file():
            raise FileNotFoundError(history_path)

        df = pd.read_csv(history_path)

        required = {
            "epoch",
            "train_mse_scaled",
            "validation_mse_scaled",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {history_path}"
            )

        plt.figure(figsize=(9, 5))

        plt.plot(
            df["epoch"],
            df["train_mse_scaled"],
            marker="o",
            linewidth=2,
            label="Training MSE",
        )

        plt.plot(
            df["epoch"],
            df["validation_mse_scaled"],
            marker="s",
            linewidth=2,
            label="Validation MSE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE (scaled RUL)")

        plt.title(
            f"{model.upper()} - Training vs Validation"
        )

        plt.legend()
        plt.grid(alpha=0.3)

        save_chart(
            chart_dir
            / f"{model}_training_history.png"
        )


def plot_model_comparison(
    run_dir: Path,
    chart_dir: Path,
) -> None:
    """Compare overall MAE and RMSE."""

    comparison_path = (
        run_dir
        / "evaluation"
        / "comparison.csv"
    )

    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)

    df = pd.read_csv(comparison_path)

    for metric, label in (
        ("mae_minutes", "MAE"),
        ("rmse_minutes", "RMSE"),
    ):

        plt.figure(figsize=(7, 5))

        bars = plt.bar(
            df["model"],
            df[metric],
        )

        plt.bar_label(
            bars,
            fmt="%.1f",
            padding=3,
        )

        plt.xlabel("Model")
        plt.ylabel("Error (minutes)")

        plt.title(
            f"Test {label}: LSTM vs GRU"
        )

        plt.ylim(
            0,
            df[metric].max() * 1.2,
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        save_chart(
            chart_dir
            / f"{label.lower()}_comparison.png"
        )


def plot_per_bearing_mae(
    run_dir: Path,
    chart_dir: Path,
) -> None:
    """Compare MAE for each test bearing."""

    metrics_path = (
        run_dir
        / "evaluation"
        / "per_bearing_metrics.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)

    pivot = df.pivot(
        index="bearing_id",
        columns="model",
        values="mae_minutes",
    )

    pivot.index = [
        name.split("/")[-1]
        for name in pivot.index
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
        rot=0,
    )

    ax.set_xlabel("Test Bearing")
    ax.set_ylabel("MAE (minutes)")

    ax.set_title(
        "Per-Bearing Test MAE: LSTM vs GRU"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    save_chart(
        chart_dir / "per_bearing_mae.png"
    )


def plot_rul_predictions(
    run_dir: Path,
    chart_dir: Path,
) -> None:
    """Plot actual and predicted RUL for each bearing."""

    for model in ("lstm", "gru"):

        prediction_path = (
            run_dir
            / "evaluation"
            / f"{model}_predictions.csv"
        )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                prediction_path
            )

        df = pd.read_csv(prediction_path)

        required = {
            "bearing_id",
            "target_step",
            "actual_rul_minutes",
            "predicted_rul_minutes",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {prediction_path}: "
                f"{required - set(df.columns)}"
            )

        for bearing_id, group in df.groupby(
            "bearing_id"
        ):

            group = group.sort_values(
                "target_step"
            )

            plt.figure(figsize=(10, 5))

            plt.plot(
                group["target_step"],
                group["actual_rul_minutes"],
                linewidth=2,
                label="Actual RUL",
            )

            plt.plot(
                group["target_step"],
                group["predicted_rul_minutes"],
                linewidth=2,
                label=f"{model.upper()} Predicted RUL",
            )

            plt.xlabel("Snapshot Index")

            plt.ylabel(
                "Remaining Useful Life (minutes)"
            )

            plt.title(
                f"{model.upper()} - {bearing_id}"
            )

            plt.legend()
            plt.grid(alpha=0.3)

            bearing_name = (
                bearing_id.split("/")[-1]
            )

            save_chart(
                chart_dir
                / f"{model}_{bearing_name}_rul.png"
            )


def main() -> None:
    """Run the complete visualization pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate charts for LSTM and GRU "
            "bearing RUL prediction."
        )
    )

    parser.add_argument(
        "--run-name",
        default="main",
        help="Name of the experiment.",
    )

    args = parser.parse_args()

    if not args.run_name or not all(
        char.isalnum() or char in "-_"
        for char in args.run_name
    ):
        raise ValueError(
            "Invalid run name."
        )

    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / args.run_name
    )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {run_dir}"
        )

    chart_dir = run_dir / "charts"

    chart_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("BEARING RUL VISUALIZATION")
    print("=" * 60)

    print("\n[1/4] Creating training charts...")

    plot_training_history(
        run_dir,
        chart_dir,
    )

    print("\n[2/4] Creating MAE/RMSE charts...")

    plot_model_comparison(
        run_dir,
        chart_dir,
    )

    print("\n[3/4] Creating per-bearing chart...")

    plot_per_bearing_mae(
        run_dir,
        chart_dir,
    )

    print("\n[4/4] Creating RUL prediction charts...")

    plot_rul_predictions(
        run_dir,
        chart_dir,
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)

    print(f"Charts saved to: {chart_dir}")


if __name__ == "__main__":
    main()