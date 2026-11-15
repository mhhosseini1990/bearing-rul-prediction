"""Replay transparent policies from saved fixed48 direct-RUL predictions."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bearing_rul.maintenance import CostScenario, ReplayPolicy, replay_strategies  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "outputs/final/model/fixed48_direct_final/predictions_raw.csv")
    parser.add_argument("--model", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    parser.add_argument("--scheduled-age", type=float, default=120)
    parser.add_argument("--warning-threshold", type=float, default=120)
    parser.add_argument("--required-lead", type=float, default=30)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final/maintenance_replay")
    args = parser.parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Direct-RUL predictions unavailable: {args.predictions}")
    data = pd.read_csv(args.predictions)
    data = data[(data.model == args.model) & (data.seed == args.seed)].copy()
    if data.bearing_id.nunique() != 15:
        raise ValueError("Replay requires one outer-fold trajectory for all 15 bearings")
    policy = ReplayPolicy(args.scheduled_age, args.warning_threshold,
                          args.required_lead, args.consecutive)
    costs = (CostScenario("low_failure_cost", corrective_replacement=3, failure_downtime=4),
             CostScenario("base"),
             CostScenario("high_failure_cost", corrective_replacement=8, failure_downtime=20))
    events, summary = replay_strategies(data, policy, costs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "strategy_events.csv", index=False)
    summary.to_csv(args.output_dir / "strategy_summary.csv", index=False)
    manifest = {"source": str(args.predictions), "model": args.model, "seed": args.seed,
                "policy": asdict(policy), "cost_scenarios": [asdict(cost) for cost in costs],
                "decision_inputs": ["step", "predicted_rul_minutes"],
                "ground_truth_use": "retrospective scoring only",
                "interpretation": "hypothetical historical replay; not industrial savings"}
    (args.output_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
