
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from bearing_rul.data.preprocessing import prepare_dataset
from bearing_rul.features.wavelet_features import FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data/processed/xjtu_wavelet_features.csv"
SPLIT_FILE = PROJECT_ROOT / "configs/bearing_split_v1.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs/main/preprocessing"


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    for path in (FEATURE_FILE, SPLIT_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Rebuilding feature scaler from training data...")

    partitions, scaler = prepare_dataset(
        feature_matrix_path=FEATURE_FILE,
        split_config_path=SPLIT_FILE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = OUTPUT_DIR / "feature_scaler_reconstructed.joblib"
    metadata_path = OUTPUT_DIR / "feature_scaler_metadata.json"

    if scaler_path.exists() or metadata_path.exists():
        raise FileExistsError(
            "Scaler or metadata already exists. Refusing to overwrite."
        )

    metadata = {
        "status": "reconstructed_not_historically_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "training_rows": len(partitions["train"]),
        "feature_matrix_sha256": sha256_file(FEATURE_FILE),
        "split_config_sha256": sha256_file(SPLIT_FILE),
        "scaler_type": type(scaler).__name__,
    }

    joblib.dump(scaler, scaler_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Scaler saved:", scaler_path)
    print("Metadata saved:", metadata_path)
    print("Status: reconstructed, historical equivalence not yet verified")


if __name__ == "__main__":
    main()
