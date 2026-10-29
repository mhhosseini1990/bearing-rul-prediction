"""CLI: replay maintenance alerts on previously saved test-set prediction CSVs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_rul.simulation.maintenance_replay import (  # noqa: E402
    ReplaySettings, load_prediction_files, run_replay, save_replay_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline maintenance replay (no model retraining)")
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--threshold-minutes", type=float, default=120.0)
    parser.add_argument("--consecutive-snapshots", type=int, default=3)
    parser.add_argument("--minimum-lead-minutes", type=float, default=30.0)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    settings = ReplaySettings(args.threshold_minutes, args.consecutive_snapshots, args.minimum_lead_minutes)
    eval_dir = args.evaluation_dir or ROOT / "outputs" / args.run_name / "evaluation"
    output_dir = args.output_dir or ROOT / "outputs" / args.run_name / "maintenance_replay"
    data = load_prediction_files(eval_dir / "lstm_predictions.csv", eval_dir / "gru_predictions.csv")
    events, summary, timeline = run_replay(data, settings)
    save_replay_results(events, summary, timeline, settings, output_dir)
    print("Historical replay completed on supplied prediction CSVs.")
    print(f"Settings: threshold={settings.threshold_minutes:g} min, "
          f"confirmation={settings.consecutive_snapshots} samples, "
          f"minimum useful lead={settings.minimum_lead_minutes:g} min")
    print("\nPer-bearing events:\n" + events.to_string(index=False))
    print("\nPer-model summary:\n" + summary.to_string(index=False))
    print(f"\nSaved replay CSV/JSON files to: {output_dir}")
    print("CAUTION: These are illustrative offline decisions; no factory deployment or cost saving is verified.")


if __name__ == "__main__":
    main()
