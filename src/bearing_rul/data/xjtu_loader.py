"""Load and validate XJTU-SY rolling-bearing vibration measurements.

Expected dataset structure:

XJTU-SY_Bearing_Datasets/
    35Hz12kN/
        Bearing1_1/
            1.csv
            2.csv
            ...
    37.5Hz11kN/
        Bearing2_1/
            ...
    40Hz10kN/
        Bearing3_1/
            ...

Each CSV contains two vibration channels:
    - Horizontal_vibration_signals
    - Vertical_vibration_signals

Files are ordered numerically to preserve measurement chronology.

No source data is included in the public repository.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def _natural_key(value: str) -> tuple:
    """Generate a natural sorting key for strings containing numbers."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def list_bearing_csvs(bearing_dir: str | Path) -> list[Path]:
    """Return a bearing's CSV files in numeric chronological order.

    Expected filenames:
        1.csv, 2.csv, 3.csv, ..., N.csv

    Raises:
        FileNotFoundError: If the bearing directory does not exist.
        ValueError: If files are missing, invalid, or non-consecutive.
    """
    folder = Path(bearing_dir)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Bearing directory not found: {folder}"
        )

    files = list(folder.glob("*.csv"))

    if not files:
        raise ValueError(
            f"No CSV files found in: {folder}"
        )

    try:
        files.sort(key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(
            f"CSV filenames must contain numeric snapshot IDs: {folder}"
        ) from exc

    numbers = [int(path.stem) for path in files]
    expected = list(range(1, len(numbers) + 1))

    if numbers != expected:
        raise ValueError(
            f"Missing or non-consecutive snapshot numbers in {folder}. "
            f"Found: {numbers[:10]}..."
        )

    return files


def load_vibration_csv(csv_path: str | Path) -> np.ndarray:
    """Read and validate one XJTU-SY vibration snapshot.

    Args:
        csv_path: Path to a vibration CSV file.

    Returns:
        NumPy array with shape (number_of_samples, 2).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV is empty, malformed, or contains
                    non-numeric or non-finite measurements.
    """
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Empty CSV file: {path}"
        ) from exc

    if list(frame.columns) != list(EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected CSV columns in {path}. "
            f"Expected: {list(EXPECTED_COLUMNS)}. "
            f"Found: {list(frame.columns)}"
        )

    if frame.empty:
        raise ValueError(
            f"CSV contains no vibration measurements: {path}"
        )

    try:
        values = frame.apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float64)

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Non-numeric vibration values in: {path}"
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            f"NaN or infinite vibration values in: {path}"
        )

    return values


def list_measurements(dataset_root: str | Path) -> pd.DataFrame:
    """Build a measurement manifest for the complete XJTU-SY dataset.

    Expected structure:
        <dataset_root>/<condition>/<bearing>/<number>.csv

    Each bearing is processed independently to preserve chronology.

    Args:
        dataset_root: Root directory containing operating conditions.

    Returns:
        Pandas DataFrame with columns:
            bearing_id
            bearing_name
            operating_condition
            filename
            path
            step

    The step index starts at zero for each bearing.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        ValueError: If the dataset structure or snapshot numbering
                    is invalid.
    """
    root = Path(dataset_root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    rows = []

    conditions = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: _natural_key(path.name),
    )

    for condition_dir in conditions:

        bearings = sorted(
            (
                path
                for path in condition_dir.iterdir()
                if path.is_dir()
            ),
            key=lambda path: _natural_key(path.name),
        )

        for bearing_dir in bearings:

            csv_files = list_bearing_csvs(bearing_dir)

            relative_bearing = (
                bearing_dir.relative_to(root)
                .as_posix()
            )

            for step, csv_path in enumerate(csv_files):

                rows.append(
                    {
                        "bearing_id": relative_bearing,
                        "bearing_name": bearing_dir.name,
                        "operating_condition": condition_dir.name,
                        "filename": csv_path.name,
                        "path": str(csv_path.resolve()),
                        "step": step,
                    }
                )

    if not rows:
        raise ValueError(
            f"No bearing measurements found under: {root}"
        )

    manifest = pd.DataFrame(rows)

    return manifest