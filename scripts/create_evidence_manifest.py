
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs/bearing_split_v1.yaml"


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieve current Git HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "git_commit_sha_unavailable"


def get_environment_metadata() -> dict:
    """Collect Python and core package versions."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def inspect_artifact(file_path: Path) -> dict:
    """Inspect file existence, size, and checksum hash without embedding raw contents."""
    if not file_path.exists():
        return {"exists": False, "status": "missing"}

    size_bytes = file_path.stat().st_size
    return {
        "exists": True,
        "status": "valid",
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
        "sha256": compute_sha256(file_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exhaustive baseline experiment evidence manifest."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="main",
        help="Name of the experiment run folder under outputs/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = OUTPUTS_DIR / args.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    training_dir = run_dir / "training"
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    replay_dir = run_dir / "policy_comparison"

    # Read training summary
    summary_file = training_dir / "training_summary.json"
    training_summary_data = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                training_summary_data = json.load(f)
        except Exception as exc:
            training_summary_data = {"error": f"Failed to parse summary: {str(exc)}"}

    # Tracked artifacts list
    tracked_files = {
        "checkpoints": {
            "best_lstm": checkpoint_dir / "best_lstm.pt",
            "best_gru": checkpoint_dir / "best_gru.pt",
        },
        "training_artifacts": {
            "training_summary": summary_file,
            "lstm_history": training_dir / "lstm_history.csv",
            "gru_history": training_dir / "gru_history.csv",
        },
        "evaluation_artifacts": {
            "comparison": eval_dir / "comparison.csv",
            "per_bearing_metrics": eval_dir / "per_bearing_metrics.csv",
            "lstm_predictions": eval_dir / "lstm_predictions.csv",
            "gru_predictions": eval_dir / "gru_predictions.csv",
        },
        "replay_artifacts": {
            "policy_summary_csv": replay_dir / "policy_comparison_summary.csv",
            "policy_summary_json": replay_dir / "policy_comparison_summary.json",
        },
    }

    file_manifest = {}
    missing_artifacts = []
    valid_count = 0

    for category, files in tracked_files.items():
        file_manifest[category] = {}
        for key, path in files.items():
            info = inspect_artifact(path)
            rel_path = str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path)
            info["relative_path"] = rel_path
            file_manifest[category][key] = info

            if info["exists"]:
                valid_count += 1
            else:
                missing_artifacts.append(rel_path)

    # Dataset provenance and configuration metadata
    dataset_provenance = {
        "dataset_name": "XJTU-SY Rolling Element Bearing Dataset",
        "split_config_path": str(SPLIT_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "seed": training_summary_data.get("seed", 42),
        "sequence_length": training_summary_data.get("sequence_length", 20),
        "target_scale_minutes": training_summary_data.get("target_scale_minutes", None),
    }

    # Explicit list of unverified claims and non-production disclaimers
    unverified_claims = [
        "No real-time factory deployment or industrial hardware integration is verified.",
        "Hypothetical maintenance costs are illustrative parameters for scenario analysis only.",
        "Model generalization to unobserved bearing geometries or load conditions is unverified.",
    ]

    manifest = {
        "manifest_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_git_sha": get_git_commit_hash(),
        "environment": get_environment_metadata(),
        "dataset_provenance_and_config": dataset_provenance,
        "run_name": args.run_name,
        "experiment_summary": training_summary_data,
        "artifact_health": {
            "total_tracked_files": valid_count + len(missing_artifacts),
            "valid_files_count": valid_count,
            "missing_files_count": len(missing_artifacts),
            "status": "complete" if len(missing_artifacts) == 0 else "partial",
        },
        "missing_artifacts": missing_artifacts,
        "unverified_claims_and_disclaimers": unverified_claims,
        "artifacts": file_manifest,
    }

    output_path = run_dir / "evidence_manifest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("COMPLETE BASELINE EXPERIMENT EVIDENCE MANIFEST CREATED")
    print("=" * 70)
    print(f"Git SHA:             {manifest['repository_git_sha']}")
    print(f"Python Version:      {manifest['environment']['python_version']}")
    print(f"Manifest Path:       {output_path}")
    print(f"Artifact Health:     {valid_count} valid files, {len(missing_artifacts)} missing")
    print(f"Missing Artifacts:   {missing_artifacts}")
    print("=" * 70)


if __name__ == "__main__":
    main()