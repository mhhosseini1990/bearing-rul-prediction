"""Create charts for the LSTM vs GRU bearing RUL experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_chart(path):
    """Save and close the current chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.name}")


def plot_training_history(run_dir, chart_dir):
    """Plot training and validation MSE for both models."""

    for model in ("lstm", "gru"):
        path = run_dir / "training" / f"{model}_history.csv"
        df = pd.read_csv(path)

        plt.figure(figsize=(9, 5))

        plt.plot(
            df["epoch"],
            df["train_mse_scaled"],
            marker="o",
            label="Training MSE",
        )

        plt.plot(
            df["epoch"],
            df["validation_mse_scaled"],
            marker="s",
            label="Validation MSE",
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE (scaled RUL)")
        plt.title(f"{model.upper()} - Training vs Validation")
        plt.legend()
        plt.grid(alpha=0.3)

        save_chart(
            chart_dir / f"{model}_training_history.png"
        )


def plot_model_comparison(run_dir, chart_dir):
    """Compare overall MAE and RMSE."""

    df = pd.read_csv(
        run_dir / "evaluation" / "comparison.csv"
    )

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

        plt.ylabel("Error (minutes)")
        plt.title(f"Test {label}: LSTM vs GRU")
        plt.ylim(0, df[metric].max() * 1.2)
        plt.grid(axis="y", alpha=0.3)

        save_chart(
            chart_dir / f"{label.lower()}_comparison.png"
        )


def plot_per_bearing_mae(run_dir, chart_dir):
    """Compare MAE for each test bearing."""

    df = pd.read_csv(
        run_dir / "evaluation" / "per_bearing_metrics.csv"
    )

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

    ax.set_xlabel("Test bearing")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Per-Bearing Test MAE")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        chart_dir / "per_bearing_mae.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved: per_bearing_mae.png")


def plot_rul_predictions(run_dir, chart_dir):
    """Plot actual and predicted RUL for each test bearing."""

    for model in ("lstm", "gru"):

        df = pd.read_csv(
            run_dir
            / "evaluation"
            / f"{model}_predictions.csv"
        )

        required = {
            "bearing_id",
            "target_step",
            "actual_rul_minutes",
            "predicted_rul_minutes",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {model} predictions: "
                f"{required - set(df.columns)}"
            )

        for bearing_id, group in df.groupby("bearing_id"):

            group = group.sort_values("target_step")

            plt.figure(figsize=(10, 5))

            plt.plot(
                group["target_step"],
                group["actual_rul_minutes"],
                label="Actual RUL",
                linewidth=2,
            )

            plt.plot(
                group["target_step"],
                group["predicted_rul_minutes"],
                label=f"{model.upper()} Predicted RUL",
                linewidth=2,
            )

            plt.xlabel("Snapshot index")
            plt.ylabel("Remaining Useful Life (minutes)")
            plt.title(
                f"{model.upper()} - "
                f"{bearing_id}"
            )

            plt.legend()
            plt.grid(alpha=0.3)

            bearing_name = bearing_id.split("/")[-1]

            save_chart(
                chart_dir
                / f"{model}_{bearing_name}_rul.png"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="main",
    )

    args = parser.parse_args()

    if not args.run_name or not all(
        char.isalnum() or char in "-_"
        for char in args.run_name
    ):
        raise ValueError("Invalid run name.")

    run_dir = PROJECT_ROOT / "outputs" / args.run_name

    chart_dir = run_dir / "charts"

    chart_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCreating training charts...")
    plot_training_history(run_dir, chart_dir)

    print("\nCreating model comparison charts...")
    plot_model_comparison(run_dir, chart_dir)

    print("\nCreating per-bearing comparison...")
    plot_per_bearing_mae(run_dir, chart_dir)

    print("\nCreating RUL prediction charts...")
    plot_rul_predictions(run_dir, chart_dir)

    print("\nVISUALIZATION COMPLETED")
    print(f"Charts saved to: {chart_dir}")


if __name__ == "__main__":
    main()