"""Inspect downloaded file counts without training a model."""
import argparse
from bearing_rul.data.xjtu_loader import list_measurements


def main():
    parser = argparse.ArgumentParser(description="Inspect local bearing CSV files")
    parser.add_argument("--data-dir", default="data/raw/XJTU-SY")
    args = parser.parse_args()

    manifest = list_measurements(args.data_dir)
    print(f"Found {len(manifest):,} CSV recordings in {manifest['bearing_id'].nunique()} bearing folders.")
    print(manifest.groupby(["operating_condition", "bearing_id"]).size().to_string())
    print("\nIMPORTANT: Confirm condition, channel order and chronological file order.")


if __name__ == "__main__":
    main()
