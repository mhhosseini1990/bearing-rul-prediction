"""Recompute fixed48 final metrics from saved common-range predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final/model/fixed48_direct_final")
    args = parser.parse_args()
    source = args.output / "predictions_common_range.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Final predictions are unavailable: {source}. Run the final experiment first.")
    data = pd.read_csv(source)
    if (data.step < 29).any():
        raise ValueError("Primary predictions contain steps below the approved common range")
    rows = []
    for (fold, model, seed, bearing), group in data.groupby(["fold", "model", "seed", "bearing_id"]):
        error = group.predicted_rul_minutes.to_numpy() - group.rul_minutes.to_numpy()
        rows.append({"fold": fold, "model": model, "seed": seed, "bearing_id": bearing,
                     "condition": group.condition.iloc[0], "mae_minutes": float(np.abs(error).mean()),
                     "rmse_minutes": float(np.sqrt(np.square(error).mean())),
                     "evaluated_snapshots": len(group)})
    per = pd.DataFrame(rows)
    summary = per.groupby(["model", "seed"], as_index=False).agg(
        macro_mae_minutes=("mae_minutes", "mean"),
        macro_rmse_minutes=("rmse_minutes", "mean"), bearings=("bearing_id", "nunique"))
    per.to_csv(args.output / "verified_per_bearing_metrics.csv", index=False)
    summary.to_csv(args.output / "verified_summary_by_seed.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
