from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


ALLOWED_REPLAY_SPLITS = frozenset({"validation", "historical_example"})


@dataclass(frozen=True)
class ReplayPolicyConfig:
    """Documented configuration for a single maintenance replay policy."""

    policy_id: str
    rul_threshold_minutes: float
    consecutive_confirmations: int
    is_baseline: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.rul_threshold_minutes < 0:
            raise ValueError("rul_threshold_minutes must be non-negative.")
        if self.consecutive_confirmations < 1:
            raise ValueError("consecutive_confirmations must be >= 1.")


@dataclass(frozen=True)
class HypotheticalCostAssumptions:
    """Explicit hypothetical cost parameters for relative policy sensitivity comparison."""

    planned_replacement_cost: float = 1000.0
    unplanned_failure_penalty: float = 5000.0
    unused_life_cost_per_minute: float = 5.0
    max_timely_lead_minutes: float = 60.0
    currency_unit: str = "hypothetical_cost_units"
    note: str = (
        "Hypothetical cost parameters used solely for relative replay sensitivity "
        "trade-off analysis; not validated industrial financial savings."
    )

    def __post_init__(self) -> None:
        if self.planned_replacement_cost < 0 or self.unplanned_failure_penalty < 0:
            raise ValueError("Cost parameters must be non-negative.")
        if self.unused_life_cost_per_minute < 0:
            raise ValueError("unused_life_cost_per_minute must be non-negative.")
        if self.max_timely_lead_minutes <= 0:
            raise ValueError("max_timely_lead_minutes must be strictly positive.")


def build_policy_grid(
    baseline_policy: ReplayPolicyConfig,
    rul_thresholds_minutes: Sequence[float],
    consecutive_confirmations_list: Sequence[int],
) -> List[ReplayPolicyConfig]:
    """Builds a deduplicated list of policies keeping the baseline policy intact."""
    if not baseline_policy.is_baseline:
        raise ValueError("baseline_policy must have is_baseline=True.")

    policies: List[ReplayPolicyConfig] = [baseline_policy]
    seen_keys = {
        (float(baseline_policy.rul_threshold_minutes), int(baseline_policy.consecutive_confirmations))
    }

    for thresh in rul_thresholds_minutes:
        for confirm in consecutive_confirmations_list:
            key = (float(thresh), int(confirm))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            policies.append(
                ReplayPolicyConfig(
                    policy_id=f"thresh_{int(thresh)}m_confirm_{int(confirm)}",
                    rul_threshold_minutes=float(thresh),
                    consecutive_confirmations=int(confirm),
                    is_baseline=False,
                    description=(
                        f"Trigger replacement when predicted RUL <= {float(thresh):.1f} min "
                        f"for {int(confirm)} consecutive step(s)."
                    ),
                )
            )

    return policies


def _simulate_single_bearing(
    bearing_df: pd.DataFrame,
    bearing_id: str,
    policy: ReplayPolicyConfig,
    cost_assumptions: HypotheticalCostAssumptions,
    actual_col: str,
    pred_col: str,
    step_col: Optional[str],
    minutes_per_step: float,
) -> Dict[str, Any]:
    """Simulates a single replay policy on one chronologically ordered bearing trajectory."""
    ordered = (
        bearing_df.sort_values(by=step_col, kind="mergesort").reset_index(drop=True)
        if step_col and step_col in bearing_df.columns
        else bearing_df.reset_index(drop=True)
    )

    actual_min = ordered[actual_col].to_numpy(dtype=np.float64) * minutes_per_step
    pred_min = ordered[pred_col].to_numpy(dtype=np.float64) * minutes_per_step
    n_steps = len(ordered)

    # Identify failure step (first point where actual_rul <= 0, else end of trajectory)
    failure_indices = np.where(actual_min <= 0.0)[0]
    failure_step = int(failure_indices[0]) if failure_indices.size > 0 else n_steps - 1

    streak = 0
    triggered = False
    replacement_step: Optional[int] = None
    useful_lead_time_minutes = 0.0

    for idx in range(n_steps):
        # Cannot execute a preventive replacement at or after physical failure
        if actual_min[idx] <= 0.0:
            break

        if pred_min[idx] <= policy.rul_threshold_minutes:
            streak += 1
        else:
            streak = 0

        if streak >= policy.consecutive_confirmations:
            triggered = True
            replacement_step = int(idx)
            useful_lead_time_minutes = float(actual_min[idx])
            break

    if not triggered or useful_lead_time_minutes <= 0.0:
        event_classification = "unplanned_failure"
        hypothetical_cost = float(cost_assumptions.unplanned_failure_penalty)
    elif useful_lead_time_minutes <= cost_assumptions.max_timely_lead_minutes:
        event_classification = "timely_replacement"
        hypothetical_cost = float(
            cost_assumptions.planned_replacement_cost
            + useful_lead_time_minutes * cost_assumptions.unused_life_cost_per_minute
        )
    else:
        event_classification = "premature_replacement"
        hypothetical_cost = float(
            cost_assumptions.planned_replacement_cost
            + useful_lead_time_minutes * cost_assumptions.unused_life_cost_per_minute
        )

    replacement_elapsed_minutes = (
        float(replacement_step * minutes_per_step) if replacement_step is not None else None
    )
    failure_elapsed_minutes = float(failure_step * minutes_per_step)

    return {
        "policy_id": policy.policy_id,
        "is_baseline": policy.is_baseline,
        "rul_threshold_minutes": float(policy.rul_threshold_minutes),
        "consecutive_confirmations": int(policy.consecutive_confirmations),
        "bearing_id": bearing_id,
        "triggered": triggered,
        "replacement_step": replacement_step,
        "replacement_elapsed_minutes": replacement_elapsed_minutes,
        "failure_step": failure_step,
        "failure_elapsed_minutes": failure_elapsed_minutes,
        "useful_lead_time_minutes": useful_lead_time_minutes,
        "event_classification": event_classification,
        "hypothetical_cost": hypothetical_cost,
    }


def evaluate_replay_policy(
    df: pd.DataFrame,
    policy: ReplayPolicyConfig,
    cost_assumptions: Optional[HypotheticalCostAssumptions] = None,
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    pred_col: str = "predicted_rul",
    step_col: Optional[str] = "window_idx",
    minutes_per_step: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates a single replay policy across all bearings in the provided dataset."""
    required = {bearing_col, actual_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory DataFrame missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Trajectory DataFrame is empty.")
    if minutes_per_step <= 0:
        raise ValueError("minutes_per_step must be strictly positive.")

    costs = cost_assumptions or HypotheticalCostAssumptions()
    per_bearing_records: List[Dict[str, Any]] = []

    for bearing_id, group in df.groupby(bearing_col, sort=True):
        record = _simulate_single_bearing(
            bearing_df=group,
            bearing_id=str(bearing_id),
            policy=policy,
            cost_assumptions=costs,
            actual_col=actual_col,
            pred_col=pred_col,
            step_col=step_col,
            minutes_per_step=minutes_per_step,
        )
        per_bearing_records.append(record)

    res_df = pd.DataFrame(per_bearing_records)
    timely_count = int((res_df["event_classification"] == "timely_replacement").sum())
    premature_count = int((res_df["event_classification"] == "premature_replacement").sum())
    failure_count = int((res_df["event_classification"] == "unplanned_failure").sum())

    triggered_rows = res_df[res_df["triggered"]]
    mean_replacement_step = (
        float(triggered_rows["replacement_step"].mean()) if not triggered_rows.empty else None
    )

    summary = {
        "policy_id": policy.policy_id,
        "is_baseline": policy.is_baseline,
        "description": policy.description,
        "rul_threshold_minutes": float(policy.rul_threshold_minutes),
        "consecutive_confirmations": int(policy.consecutive_confirmations),
        "n_bearings": int(len(res_df)),
        "timely_replacement_count": timely_count,
        "premature_replacement_count": premature_count,
        "unplanned_failure_count": failure_count,
        "mean_useful_lead_time_minutes": float(res_df["useful_lead_time_minutes"].mean()),
        "median_useful_lead_time_minutes": float(res_df["useful_lead_time_minutes"].median()),
        "mean_replacement_step": mean_replacement_step,
        "total_hypothetical_cost": float(res_df["hypothetical_cost"].sum()),
        "mean_hypothetical_cost_per_bearing": float(res_df["hypothetical_cost"].mean()),
    }

    return {
        "summary": summary,
        "per_bearing": per_bearing_records,
    }


def run_replay_sensitivity_sweep(
    df: pd.DataFrame,
    dataset_split: str,
    baseline_policy: ReplayPolicyConfig,
    candidate_policies: Iterable[ReplayPolicyConfig],
    cost_assumptions: Optional[HypotheticalCostAssumptions] = None,
    bearing_col: str = "bearing_id",
    actual_col: str = "actual_rul",
    pred_col: str = "predicted_rul",
    step_col: Optional[str] = "window_idx",
    minutes_per_step: float = 1.0,
) -> Dict[str, Any]:
    """Compares candidate replay policies against an intact baseline on validation/historical data.

    Explicitly rejects held-out test splits so policy settings are never tuned on test outcomes.
    """
    normalized_split = dataset_split.strip().lower()
    if normalized_split not in ALLOWED_REPLAY_SPLITS:
        raise ValueError(
            f"Unsupported dataset_split '{dataset_split}'. Policy sensitivity sweeps are "
            f"restricted to {sorted(ALLOWED_REPLAY_SPLITS)} to prevent held-out test leakage."
        )
    if not baseline_policy.is_baseline:
        raise ValueError("baseline_policy must have is_baseline=True.")

    costs = cost_assumptions or HypotheticalCostAssumptions()

    # 1. Evaluate baseline policy first and preserve its outputs untouched
    baseline_eval = evaluate_replay_policy(
        df=df,
        policy=baseline_policy,
        cost_assumptions=costs,
        bearing_col=bearing_col,
        actual_col=actual_col,
        pred_col=pred_col,
        step_col=step_col,
        minutes_per_step=minutes_per_step,
    )
    baseline_summary = dict(baseline_eval["summary"])
    baseline_summary["cost_delta_vs_baseline"] = 0.0
    baseline_summary["lead_time_delta_vs_baseline_minutes"] = 0.0

    policy_summaries: List[Dict[str, Any]] = [baseline_summary]
    all_per_bearing: List[Dict[str, Any]] = list(baseline_eval["per_bearing"])

    # 2. Evaluate candidate policies without mutating baseline results
    for policy in candidate_policies:
        if policy.policy_id == baseline_policy.policy_id:
            continue
        if policy.is_baseline:
            raise ValueError(
                f"Duplicate baseline flag on candidate policy '{policy.policy_id}'."
            )

        cand_eval = evaluate_replay_policy(
            df=df,
            policy=policy,
            cost_assumptions=costs,
            bearing_col=bearing_col,
            actual_col=actual_col,
            pred_col=pred_col,
            step_col=step_col,
            minutes_per_step=minutes_per_step,
        )
        cand_summary = dict(cand_eval["summary"])
        cand_summary["cost_delta_vs_baseline"] = float(
            cand_summary["total_hypothetical_cost"] - baseline_summary["total_hypothetical_cost"]
        )
        cand_summary["lead_time_delta_vs_baseline_minutes"] = float(
            cand_summary["mean_useful_lead_time_minutes"]
            - baseline_summary["mean_useful_lead_time_minutes"]
        )

        policy_summaries.append(cand_summary)
        all_per_bearing.extend(cand_eval["per_bearing"])

    return {
        "dataset_split": normalized_split,
        "minutes_per_step": float(minutes_per_step),
        "hypothetical_cost_assumptions": asdict(costs),
        "scope_and_limitations": {
            "held_out_test_used": False,
            "validated_industrial_savings_claimed": False,
            "statement": (
                f"Replay comparison executed strictly on '{normalized_split}' trajectories. "
                "Cost outputs reflect hypothetical sensitivity assumptions and are not "
                "validated industrial savings."
            ),
        },
        "baseline_policy": baseline_eval,
        "policy_comparison": policy_summaries,
        "per_bearing_events": all_per_bearing,
    }


def export_replay_sensitivity_report(
    report: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "replay_sensitivity",
) -> None:
    """Exports the replay sensitivity JSON summary and CSV comparison tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(report["policy_comparison"]).to_csv(
        out_dir / f"{prefix}_policy_comparison.csv", index=False
    )
    pd.DataFrame(report["per_bearing_events"]).to_csv(
        out_dir / f"{prefix}_per_bearing_events.csv", index=False
    )