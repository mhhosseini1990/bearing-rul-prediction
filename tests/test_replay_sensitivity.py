import json
from pathlib import Path
import pandas as pd
import pytest

from bearing_rul.simulation.replay_sensitivity import (
    HypotheticalCostAssumptions,
    ReplayPolicyConfig,
    build_policy_grid,
    export_replay_sensitivity_report,
    run_replay_sensitivity_sweep,
)


@pytest.fixture
def validation_trajectories() -> pd.DataFrame:
    """Creates labeled validation trajectories demonstrating timely, premature, and missed triggers."""
    # Bearing_1: Transient dip at window 2 (actual_rul=70), sustained drop starting window 4 (actual_rul=50)
    b1 = pd.DataFrame(
        {
            "bearing_id": ["Bearing_1"] * 10,
            "window_idx": list(range(10)),
            "actual_rul": [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "predicted_rul": [88.0, 76.0, 42.0, 64.0, 44.0, 38.0, 28.0, 19.0, 9.0, 0.0],
        }
    )
    # Bearing_2: Smooth countdown; triggers late or misses if threshold is too low
    b2 = pd.DataFrame(
        {
            "bearing_id": ["Bearing_2"] * 6,
            "window_idx": list(range(6)),
            "actual_rul": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "predicted_rul": [55.0, 45.0, 35.0, 25.0, 15.0, 0.0],
        }
    )
    return pd.concat([b1, b2], ignore_index=True)


def test_threshold_and_confirmation_sweep_classifications(validation_trajectories: pd.DataFrame):
    """Verify threshold & consecutive confirmation variations produce expected timings and events."""
    baseline = ReplayPolicyConfig(
        policy_id="baseline_45m_1step",
        rul_threshold_minutes=45.0,
        consecutive_confirmations=1,
        is_baseline=True,
        description="Baseline single-step trigger at 45 minutes.",
    )
    candidates = build_policy_grid(
        baseline_policy=baseline,
        rul_thresholds_minutes=[45.0, 10.0],
        consecutive_confirmations_list=[1, 2],
    )
    costs = HypotheticalCostAssumptions(
        planned_replacement_cost=1000.0,
        unplanned_failure_penalty=5000.0,
        unused_life_cost_per_minute=5.0,
        max_timely_lead_minutes=60.0,
    )

    report = run_replay_sensitivity_sweep(
        df=validation_trajectories,
        dataset_split="validation",
        baseline_policy=baseline,
        candidate_policies=candidates,
        cost_assumptions=costs,
    )

    # Baseline must remain completely intact
    base_summary = report["baseline_policy"]["summary"]
    assert base_summary["policy_id"] == "baseline_45m_1step"
    assert base_summary["is_baseline"] is True

    # Under baseline (thresh=45, confirm=1):
    # Bearing_1 triggers on transient dip at window 2 (lead=70 > 60 -> premature_replacement)
    # Bearing_2 triggers at window 1 (lead=40 <= 60 -> timely_replacement)
    assert base_summary["premature_replacement_count"] == 1
    assert base_summary["timely_replacement_count"] == 1
    assert base_summary["unplanned_failure_count"] == 0

    # Under thresh=45, confirm=2:
    # Bearing_1 ignores transient dip at window 2 and triggers at window 5 (lead=40 <= 60 -> timely_replacement)
    # Bearing_2 triggers at window 2 (lead=30 <= 60 -> timely_replacement)
    comp_df = pd.DataFrame(report["policy_comparison"])
    confirm2_row = comp_df[comp_df["policy_id"] == "thresh_45m_confirm_2"].iloc[0]
    assert confirm2_row["timely_replacement_count"] == 2
    assert confirm2_row["premature_replacement_count"] == 0
    assert confirm2_row["unplanned_failure_count"] == 0
    assert confirm2_row["cost_delta_vs_baseline"] < 0.0

    # Under thresh=10, confirm=2:
    # Neither bearing reaches 2 consecutive steps <= 10 before actual_rul=0 -> 2 unplanned_failures
    strict_row = comp_df[comp_df["policy_id"] == "thresh_10m_confirm_2"].iloc[0]
    assert strict_row["unplanned_failure_count"] == 2
    assert strict_row["mean_useful_lead_time_minutes"] == 0.0


def test_blocks_held_out_test_split_and_records_assumptions(
    validation_trajectories: pd.DataFrame, tmp_path: Path
):
    """Verify held-out test splits are rejected and hypothetical cost/scope notes are recorded."""
    baseline = ReplayPolicyConfig(
        policy_id="baseline",
        rul_threshold_minutes=30.0,
        consecutive_confirmations=1,
        is_baseline=True,
    )

    with pytest.raises(ValueError, match="prevent held-out test leakage"):
        run_replay_sensitivity_sweep(
            df=validation_trajectories,
            dataset_split="held_out_test",
            baseline_policy=baseline,
            candidate_policies=[],
        )

    report = run_replay_sensitivity_sweep(
        df=validation_trajectories,
        dataset_split="historical_example",
        baseline_policy=baseline,
        candidate_policies=[],
    )

    assert report["scope_and_limitations"]["held_out_test_used"] is False
    assert report["scope_and_limitations"]["validated_industrial_savings_claimed"] is False
    assert "hypothetical" in report["hypothetical_cost_assumptions"]["note"].lower()

    export_replay_sensitivity_report(report, tmp_path / "replay_out")
    assert (tmp_path / "replay_out" / "replay_sensitivity_report.json").exists()
    assert (tmp_path / "replay_out" / "replay_sensitivity_policy_comparison.csv").exists()
    assert (tmp_path / "replay_out" / "replay_sensitivity_per_bearing_events.csv").exists()

    loaded = json.loads(
        (tmp_path / "replay_out" / "replay_sensitivity_report.json").read_text(encoding="utf-8")
    )
    assert loaded["dataset_split"] == "historical_example"