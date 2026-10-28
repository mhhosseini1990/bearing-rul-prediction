"""Build a wavelet feature matrix from XJTU-SY bearing measurements."""

from pathlib import Path

import numpy as np
import pandas as pd

from bearing_rul.data.xjtu_loader import (
    list_measurements,
    load_vibration_csv,
)
from bearing_rul.data.rul_labels import add_remaining_recordings
from bearing_rul.features.wavelet_features import (
    FEATURE_NAMES,
    extract_wavelet_features,
)


def build_feature_matrix(dataset_root: str | Path) -> pd.DataFrame:
    """Build one feature row per chronological vibration snapshot.

    The resulting DataFrame contains:
        bearing_id, bearing_name, operating_condition,
        filename, step, rul_remaining_recordings, rul_minutes,
        and eight wavelet features.

    Each snapshot is processed independently.
    """
    manifest = list_measurements(dataset_root)
    manifest = add_remaining_recordings(manifest)

    rows = []

    for row in manifest.itertuples(index=False):
        signal = load_vibration_csv(Path(row.path))
        features = extract_wavelet_features(signal)

        record = {
            "bearing_id": row.bearing_id,
            "bearing_name": row.bearing_name,
            "operating_condition": row.operating_condition,
            "filename": row.filename,
            "step": row.step,
            "rul_remaining_recordings": row.rul_remaining_recordings,
            "rul_minutes": row.rul_minutes,
        }

        record.update(
            dict(zip(FEATURE_NAMES, features, strict=True))
        )

        rows.append(record)

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("Feature matrix is empty")

    if result.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Duplicate bearing/step pairs found")

    feature_values = result.loc[:, FEATURE_NAMES].to_numpy()

    if not np.isfinite(feature_values).all():
        raise ValueError("Feature matrix contains non-finite values")

    return result