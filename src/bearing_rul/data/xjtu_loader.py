"""Find XJTU-SY-like bearing CSVs in nested directories.

No source data is included in the public repository.
Verify the actual layout and sampling protocol against dataset documentation.
"""
from pathlib import Path
import re

import pandas as pd


def _natural_key(value: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", value))


def list_measurements(dataset_root: str | Path) -> pd.DataFrame:
    """Return one row per measurement, ordered within each bearing.

    Expected minimum layout: <root>/<condition>/<bearing>/<number>.csv.
    Other nested layouts are also discovered, but metadata should be checked.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    csv_files = sorted(root.rglob("*.csv"), key=lambda p: _natural_key(str(p.relative_to(root))))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under {root}")

    rows = []
    for path in csv_files:
        parts = path.relative_to(root).parts
        bearing_name = parts[-2] if len(parts) >= 2 else "unknown"
        condition = parts[-3] if len(parts) >= 3 else "unknown"
        # Include the relative bearing folder to avoid identical names colliding.
        relative_bearing = str(path.parent.relative_to(root)).replace("\\", "/")
        rows.append({
            "bearing_id": relative_bearing,
            "bearing_name": bearing_name,
            "operating_condition": condition,
            "filename": path.name,
            "path": str(path.resolve()),
        })

    # Sort by bearing and natural file name (1.csv, 2.csv, 10.csv).
    rows.sort(key=lambda r: (_natural_key(r["bearing_id"]), _natural_key(r["filename"])))
    manifest = pd.DataFrame(rows)
    manifest["step"] = manifest.groupby("bearing_id").cumcount()
    return manifest
