"""Train LSTM and GRU models on XJTU-SY wavelet features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Allow execution directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from bearing_rul.data.preprocessing import prepare_dataset
from bearing_rul.data.sequences import build_sequences
from bearing_rul.models.lstm import LSTMRegressor
from bearing_rul.models.gru import GRURegressor
from bearing_rul.training.dataset import prepare_rul_dataloaders
from bearing_rul.training.trainer import set_seed, train_model


FEATURE_FILE = (
    PROJECT_ROOT / "data/processed/xjtu_wavelet_features.csv"
)

SPLIT_CONFIG = (
    PROJECT_ROOT / "configs/bearing_split_v1.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train bearing RUL prediction models."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="main",
    )

    return parser.parse_args()


def get_device(requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def prepare_data(sequence_length, batch_size):
    """Load, split, scale and sequence the real dataset."""

    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    if not SPLIT_CONFIG.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {SPLIT_CONFIG}"
        )

    print("\n[1/5] Loading and preprocessing dataset...")

    partitions, feature_scaler = prepare_dataset(
    feature_matrix_path=FEATURE_FILE,
    split_config_path=SPLIT_CONFIG,
    )

    required = {"train", "validation", "test"}

    if not required.issubset(partitions):
        raise ValueError(
            "Expected train, validation and test partitions. "
            f"Found: {list(partitions.keys())}"
        )

    print("\n[2/5] Building temporal sequences...")

    sequence_data = {}

    for partition_name in ("train", "validation", "test"):

        X, y, metadata = build_sequences(
            partitions[partition_name],
            sequence_length=sequence_length,
            stride=1,
        )

        if X.ndim != 3 or X.shape[2] != 8:
            raise ValueError(
                f"Unexpected feature shape for {partition_name}: "
                f"{X.shape}"
            )

        sequence_data[partition_name] = {
            "X": X,
            "y": y,
            "metadata": metadata,
        }

        print(
            f"{partition_name:>10}: "
            f"X={X.shape}, y={y.shape}"
        )

    print("\n[3/5] Preparing target scaling and DataLoaders...")

    loaders = prepare_rul_dataloaders(
        sequence_data["train"]["X"],
        sequence_data["train"]["y"],
        sequence_data["validation"]["X"],
        sequence_data["validation"]["y"],
        sequence_data["test"]["X"],
        sequence_data["test"]["y"],
        batch_size=batch_size,
    )

    target_scale = float(
        loaders["target_scaler"].scale_
    )

    print(f"Training target scale: {target_scale:.2f} minutes")

    return loaders, target_scale, sequence_data


def save_history(history, output_path):
    """Save training and validation losses as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_frame = pd.DataFrame({
        "epoch": np.arange(
            1,
            len(history["train_loss"]) + 1,
        ),
        "train_mse_scaled": history["train_loss"],
        "validation_mse_scaled": history["val_loss"],
    })

    history_frame.to_csv(
        output_path,
        index=False,
    )


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive.")

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    if args.sequence_length < 1:
        raise ValueError("sequence-length must be positive.")

    if args.hidden_size < 1:
        raise ValueError("hidden-size must be positive.")

    if args.patience < 1:
        raise ValueError("patience must be positive.")

    if not args.run_name or not all(
        char.isalnum() or char in "-_"
        for char in args.run_name
    ):
        raise ValueError(
            "run-name must contain only letters, "
            "numbers, hyphens or underscores."
        )

    set_seed(args.seed)

    device = get_device(args.device)

    print("=" * 60)
    print("ROLLING BEARING RUL PREDICTION")
    print("LSTM vs GRU - Training Experiment")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Seed: {args.seed}")
    print(f"Run name: {args.run_name}")

    loaders, target_scale, sequence_data = prepare_data(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )

    checkpoint_dir = (
        OUTPUT_DIR / args.run_name / "checkpoints"
    )

    training_dir = (
        OUTPUT_DIR / args.run_name / "training"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_classes = {
        "lstm": LSTMRegressor,
        "gru": GRURegressor,
    }

    summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "target_scale_minutes": target_scale,
        "sequence_counts": {
            name: int(len(data["y"]))
            for name, data in sequence_data.items()
        },
        "models": {},
    }

    print("\n[4/5] Training LSTM and GRU...")

    for model_name, model_class in model_classes.items():

        # Reset the seed to make initialization reproducible.
        set_seed(args.seed)

        print("\n" + "=" * 60)
        print(f"TRAINING {model_name.upper()}")
        print("=" * 60)

        model = model_class(
            input_size=8,
            hidden_size=args.hidden_size,
            num_layers=1,
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"Trainable parameters: {parameter_count}")

        checkpoint_path = (
            checkpoint_dir / f"best_{model_name}.pt"
        )

        result = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["validation"],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            device=device,
            checkpoint_path=checkpoint_path,
            target_scale=target_scale,
        )

        history_path = (
            training_dir / f"{model_name}_history.csv"
        )

        save_history(
            result["history"],
            history_path,
        )

        summary["models"][model_name] = {
            "parameter_count": parameter_count,
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "best_validation_mse_scaled": (
                result["best_val_loss"]
            ),
            "checkpoint_path": str(
                checkpoint_path.relative_to(PROJECT_ROOT)
            ),
            "history_path": str(
                history_path.relative_to(PROJECT_ROOT)
            ),
        }

        print(
            f"\n{model_name.upper()} completed."
            f"\nBest epoch: {result['best_epoch']}"
            f"\nBest validation MSE (scaled): "
            f"{result['best_val_loss']:.6f}"
        )

    print("\n[5/5] Saving training summary...")

    summary_path = (
        training_dir / "training_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    for name, details in summary["models"].items():
        print(
            f"{name.upper()}: "
            f"best epoch={details['best_epoch']}, "
            f"validation MSE="
            f"{details['best_validation_mse_scaled']:.6f}"
        )

    print(f"\nSummary saved to: {summary_path}")

    print(
        "\nNote: Test predictions and final MAE/RMSE "
        "have not been calculated."
    )


if __name__ == "__main__":
    main()