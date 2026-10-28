"""Validate all XJTU-SY vibration CSV files."""

from pathlib import Path
import csv
import sys

from bearing_rul.data.xjtu_loader import (
    list_bearing_csvs,
    load_vibration_csv,
)


DATASET_ROOT = Path(
    "data/raw/XJTU-SY_Bearing_Datasets"
)

EXPECTED_SAMPLES = 32768

EXPECTED_CHANNELS = 2


def main():
    total_files = 0
    total_bearings = 0
    errors = []

    for condition_dir in sorted(DATASET_ROOT.iterdir()):

        if not condition_dir.is_dir():
            continue

        for bearing_dir in sorted(condition_dir.iterdir()):

            if not bearing_dir.is_dir():
                continue

            total_bearings += 1

            try:
                files = list_bearing_csvs(bearing_dir)

            except (ValueError, FileNotFoundError) as exc:
                errors.append(
                    (str(bearing_dir), str(exc))
                )
                continue

            print(
                f"Checking {condition_dir.name}/"
                f"{bearing_dir.name}: {len(files)} files"
            )

            for path in files:
                total_files += 1

                try:
                    values = load_vibration_csv(path)

                    if values.shape != (
                        EXPECTED_SAMPLES,
                        EXPECTED_CHANNELS,
                    ):
                        raise ValueError(
                            f"Unexpected shape: {values.shape}"
                        )

                except (ValueError, OSError) as exc:
                    errors.append(
                        (str(path), str(exc))
                    )

    print("\n===== VALIDATION SUMMARY =====")
    print(f"Bearings checked: {total_bearings}")
    print(f"CSV files checked: {total_files}")
    print(f"Errors found: {len(errors)}")

    if errors:
        print("\nFirst 20 errors:")
        for path, error in errors[:20]:
            print(f"{path}: {error}")
        sys.exit(1)

    print("All checked CSV files passed validation.")


if __name__ == "__main__":
    main()