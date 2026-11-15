r"""Build and verify the single XJTU-SY fixed-48 feature matrix.

Use only ``bearing_rul.features.fixed_features`` for extraction.  All 48
features are computed per snapshot without selecting or fitting on labels.

Commands (from the repository root)::

    .\.venv\Scripts\python.exe scripts\prepare_data.py --extract
    .\.venv\Scripts\python.exe scripts\prepare_data.py --validate

The existing matrix is never overwritten unless ``--overwrite`` is supplied.
This script does not delete old experiment data or change model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_rul.data.xjtu_loader import list_measurements, load_vibration_csv
from bearing_rul.features.fixed_features import (
    DEFAULT_SAMPLING_FREQUENCY_HZ,
    FEATURE_NAMES,
    FEATURE_SET_SHA256,
    FEATURE_SET_VERSION,
    extract_fixed_features,
)

RAW_DIR = ROOT / "data" / "raw" / "XJTU-SY_Bearing_Datasets"
OUTPUT = ROOT / "data" / "processed" / "xjtu_fixed_features_48.csv"
META_COLUMNS = ("bearing_id", "condition", "step", "elapsed_minutes", "rul_minutes")
ALL_COLUMNS = (*META_COLUMNS, *FEATURE_NAMES)
EXPECTED_BEARINGS = 15
EXPECTED_CONDITIONS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_bearing_structure(frame: pd.DataFrame) -> None:
    """Check row identity, timestamp order, lifetime labels, and conditions."""
    if frame.empty:
        raise ValueError("Empty data matrix")
    expected_ids = {f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6)}
    actual_ids = set(frame["bearing_id"].astype(str).unique())
    if actual_ids != expected_ids:
        raise ValueError(f"Unexpected bearing IDs. Missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}")
    if frame["condition"].nunique() != EXPECTED_CONDITIONS:
        raise ValueError(f"Expected {EXPECTED_CONDITIONS} operating conditions")
    if frame.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Duplicate bearing_id/step entries")
    per_bearing = frame.groupby("bearing_id", sort=True)
    if (per_bearing["condition"].nunique() != 1).any():
        raise ValueError("A bearing belongs to more than one operating condition")
    if (frame.groupby("condition")["bearing_id"].nunique() != 5).any():
        raise ValueError("Each condition must contain exactly five distinct bearings")
    for bearing, part in per_bearing:
        steps = part["step"].to_numpy()
        n = len(part)
        if not np.array_equal(np.sort(steps), np.arange(n)):
            raise ValueError(f"{bearing}: step indices must be 0..{n-1} without gaps")
        if not np.array_equal(part["elapsed_minutes"].to_numpy(), steps):
            raise ValueError(f"{bearing}: elapsed_minutes differs from step")
        if not np.array_equal(part["rul_minutes"].to_numpy(), n - 1 - steps):
            raise ValueError(f"{bearing}: incorrect snapshot-count RUL labels")


def _load_inventory() -> pd.DataFrame:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Missing raw XJTU-SY folder: {RAW_DIR}")
    inventory = list_measurements(RAW_DIR).copy()
    expected = {"path", "bearing_name", "operating_condition", "step"}
    missing = expected - set(inventory.columns)
    if missing:
        raise ValueError(f"Measurement inventory missing columns: {sorted(missing)}")
    if inventory.empty or inventory[list(expected)].isna().any().any():
        raise ValueError("Missing or invalid raw measurement records")
    inventory["step"] = pd.to_numeric(inventory["step"], errors="raise").astype(int)
    if inventory.duplicated(["bearing_name", "step"]).any():
        raise ValueError("Duplicate raw measurement entries")
    inventory["bearing_rows"] = inventory.groupby("bearing_name")["step"].transform("size")
    # Check inventory structure BEFORE the relatively expensive extraction.
    _check_bearing_structure(pd.DataFrame({
        "bearing_id": inventory["bearing_name"],
        "condition": inventory["operating_condition"],
        "step": inventory["step"],
        "elapsed_minutes": inventory["step"],
        "rul_minutes": inventory["bearing_rows"] - inventory["step"] - 1,
    }))
    return inventory.sort_values(["bearing_name", "step"], kind="mergesort")


def _extract_one(row: dict) -> dict:
    """Top-level worker function; safe for Windows ProcessPoolExecutor spawn."""
    signal = load_vibration_csv(Path(row["path"]))
    features = extract_fixed_features(signal, DEFAULT_SAMPLING_FREQUENCY_HZ)
    step = int(row["step"])
    return {
        "bearing_id": str(row["bearing_name"]),
        "condition": str(row["operating_condition"]),
        "step": step,
        "elapsed_minutes": step,  # XJTU-SY snapshots are recorded once per minute.
        "rul_minutes": int(row["bearing_rows"]) - step - 1,
        **features,
    }


def _validate_frame(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != ALL_COLUMNS:
        missing = [name for name in ALL_COLUMNS if name not in frame.columns]
        extra = [name for name in frame.columns if name not in ALL_COLUMNS]
        raise ValueError(f"Incorrect feature matrix columns/order. Missing={missing}, extra={extra}")
    if frame.isna().any().any():
        raise ValueError("Matrix contains missing values")
    numeric = frame.loc[:, ["step", "elapsed_minutes", "rul_minutes", *FEATURE_NAMES]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Matrix contains NaN or infinity")
    if (numeric["rul_minutes"] < 0).any():
        raise ValueError("RUL labels cannot be negative")
    _check_bearing_structure(frame)


def extract(workers: int, overwrite: bool) -> None:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if OUTPUT.exists() and not overwrite:
        raise FileExistsError(f"{OUTPUT} already exists. Use --validate or --extract --overwrite")

    inventory = _load_inventory()
    records = inventory.to_dict("records")
    print(f"Extracting {len(records)} snapshots from {EXPECTED_BEARINGS} bearings ", flush=True)
    print(f"Feature registry: {FEATURE_SET_VERSION} ({len(FEATURE_NAMES)} fixed columns)", flush=True)

    if workers == 1:
        rows = [_extract_one(row) for row in records]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_extract_one, records, chunksize=8))

    frame = pd.DataFrame(rows).loc[:, list(ALL_COLUMNS)]
    frame = frame.sort_values(["bearing_id", "step"], kind="mergesort").reset_index(drop=True)
    _validate_frame(frame)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")

    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        # A disk round trip verifies precisely what downstream training will read.
        stored = pd.read_csv(temporary)
        _validate_frame(stored)
        manifest = {
            "dataset": "XJTU-SY",
            "feature_set": FEATURE_SET_VERSION,
            "feature_registry_sha256": FEATURE_SET_SHA256,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "rows": len(stored),
            "bearings": int(stored["bearing_id"].nunique()),
            "conditions": int(stored["condition"].nunique()),
            "sampling_frequency_hz": DEFAULT_SAMPLING_FREQUENCY_HZ,
            "snapshot_interval_minutes": 1,
            "rul_definition": "last_snapshot_step - current_step",
            "sha256": _sha256(temporary),
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, OUTPUT)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

    print(f"WROTE {OUTPUT}: {len(frame)} snapshots, 15 bearings, 48 fixed features")
    print(f"CSV SHA256: {_sha256(OUTPUT)}")


def validate() -> None:
    if not OUTPUT.is_file():
        raise FileNotFoundError(f"Missing {OUTPUT}. First run: python scripts/prepare_data.py --extract")
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    frame = pd.read_csv(OUTPUT)
    _validate_frame(frame)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": "XJTU-SY",
        "feature_set": FEATURE_SET_VERSION,
        "feature_registry_sha256": FEATURE_SET_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "rows": len(frame),
        "bearings": EXPECTED_BEARINGS,
        "conditions": EXPECTED_CONDITIONS,
        "sha256": _sha256(OUTPUT),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest mismatch: {key}; actual={manifest.get(key)!r}, expected={expected!r}")
    print(f"VALID {OUTPUT.name}: {len(frame)} snapshots, 15 bearings, 3 conditions, 48 fixed features")
    print(f"Registry SHA256: {FEATURE_SET_SHA256}")
    print(f"CSV SHA256: {_sha256(OUTPUT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--extract", action="store_true", help="Extract fixed-48 features from all raw snapshots")
    action.add_argument("--validate", action="store_true", help="Validate prepared fixed-48 CSV and manifest (default)")
    parser.add_argument("--workers", type=int, default=4, help="Extraction processes; use 1 for debugging")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacement of the fixed-48 CSV")
    args = parser.parse_args()
    if args.overwrite and not args.extract:
        parser.error("--overwrite requires --extract")
    if args.extract:
        extract(args.workers, args.overwrite)
    else:
        validate()


if __name__ == "__main__":
    main()
