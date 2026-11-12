
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURE_FILE = PROJECT_ROOT / "data/processed/xjtu_wavelet_features.csv"
SPLIT_FILE = PROJECT_ROOT / "configs/bearing_split_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path: Path) -> dict:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()

    if not path.is_file():
        return {
            "relative_path": relative_path,
            "status": "missing",
            "exists": False,
        }

    return {
        "relative_path": relative_path,
        "status": "present_checksum_recorded",
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_metadata() -> dict:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git("status", "--porcelain")

    return {
        "commit_sha": run_git("rev-parse", "HEAD"),
        "working_tree_clean": status == "" if status is not None else None,
        "working_tree_status": status,
    }


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "pandas",
                "torch",
                "scikit-learn",
                "PyWavelets",
                "PyYAML",
                "joblib",
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="main")
    args = parser.parse_args()

    if (
        not args.run_name
        or args.run_name in {".", ".."}
        or Path(args.run_name).name != args.run_name
    ):
        raise ValueError("Invalid run name")

    run_dir = OUTPUTS_DIR / args.run_name

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    output_path = run_dir / "evidence_manifest.json"

    if output_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {output_path}"
        )

    training_dir = run_dir / "training"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    replay_dir = run_dir / "maintenance_replay"
    preprocessing_dir = run_dir / "preprocessing"

    tracked_files = {
        "dataset": {
            "feature_matrix": FEATURE_FILE,
            "bearing_split": SPLIT_FILE,
        },
        "preprocessing": {
            "feature_scaler": (
                preprocessing_dir
                / "feature_scaler_reconstructed.joblib"
            ),
            "feature_scaler_metadata": (
                preprocessing_dir
                / "feature_scaler_metadata.json"
            ),
        },
        "training": {
            "summary": training_dir / "training_summary.json",
            "lstm_history": training_dir / "lstm_history.csv",
            "gru_history": training_dir / "gru_history.csv",
        },
        "checkpoints": {
            "lstm": checkpoint_dir / "best_lstm.pt",
            "gru": checkpoint_dir / "best_gru.pt",
        },
        "evaluation": {
            "comparison": eval_dir / "comparison.csv",
            "per_bearing_metrics": (
                eval_dir / "per_bearing_metrics.csv"
            ),
            "lstm_predictions": (
                eval_dir / "lstm_predictions.csv"
            ),
            "gru_predictions": (
                eval_dir / "gru_predictions.csv"
            ),
        },
        "maintenance_replay": {
            "config": replay_dir / "replay_config.json",
            "events": replay_dir / "replay_events.csv",
            "summary": replay_dir / "replay_summary.csv",
            "timeline": replay_dir / "replay_timeline.csv",
        },
    }

    artifacts = {}
    missing = []

    for category, files in tracked_files.items():
        artifacts[category] = {}

        for name, path in files.items():
            info = inspect_file(path)
            artifacts[category][name] = info

            if not info["exists"]:
                missing.append(info["relative_path"])

    summary_path = training_dir / "training_summary.json"
    training_summary = None

    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as file:
            training_summary = json.load(file)

    scaler_metadata_path = (
        preprocessing_dir / "feature_scaler_metadata.json"
    )
    scaler_metadata = None

    if scaler_metadata_path.is_file():
        with scaler_metadata_path.open(encoding="utf-8") as file:
            scaler_metadata = json.load(file)

    manifest = {
        "manifest_version": "3.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_name": args.run_name,
        "git": git_metadata(),
        "environment": environment_metadata(),
        "dataset": {
            "name": "XJTU-SY",
            "split_unit": "bearing",
            "feature_matrix": artifacts["dataset"][
                "feature_matrix"
            ],
            "split_config": artifacts["dataset"][
                "bearing_split"
            ],
        },
        "training_summary": training_summary,
        "feature_scaler_metadata": scaler_metadata,
        "artifact_health": {
            "total": sum(
                len(files) for files in tracked_files.values()
            ),
            "missing_count": len(missing),
            "status": (
                "complete_file_inventory"
                if not missing
                else "incomplete_file_inventory"
            ),
        },
        "missing_artifacts": missing,
        "artifacts": artifacts,
        "limitations": [
            "Reconstructed feature scaler is not historically verified.",
            "Checksums establish current file identity, not historical equivalence.",
            "File presence does not prove scientific correctness.",
            "Industrial deployment and cost savings are not validated.",
        ],
    }

    # Exclusive creation prevents accidental overwrite.
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("Evidence manifest created:", output_path)
    print("Tracked files:", manifest["artifact_health"]["total"])
    print("Missing files:", len(missing))
    print("Inventory status:", manifest["artifact_health"]["status"])

    if missing:
        print("Missing artifacts:")
        for path in missing:
            print(" -", path)


if __name__ == "__main__":
    main()
