"""Historical replay of transparent maintenance policies.

The module never extends a trajectory after a counterfactual replacement.  It
only asks when each policy would first act on information available at that
row and scores that decision against the observed run-to-failure endpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayPolicy:
    scheduled_age_minutes: float = 120.0
    rul_warning_minutes: float = 120.0
    required_lead_minutes: float = 30.0
    consecutive_inspections: int = 3
    inspection_cadence_minutes: float = 1.0


@dataclass(frozen=True)
class CostScenario:
    name: str = "base"
    preventive_replacement: float = 1.0
    corrective_replacement: float = 5.0
    failure_downtime: float = 10.0
    premature_replacement: float = 1.0


def _persistent_first(mask: np.ndarray, n: int) -> int | None:
    if n < 1:
        raise ValueError("consecutive_inspections must be positive")
    hits = np.convolve(mask.astype(int), np.ones(n, dtype=int), mode="valid")
    indices = np.flatnonzero(hits == n)
    return int(indices[0] + n - 1) if len(indices) else None


def _decision_row(group: pd.DataFrame, strategy: str, policy: ReplayPolicy) -> int | None:
    if strategy == "scheduled":
        candidates = np.flatnonzero(group["step"].to_numpy(float) >= policy.scheduled_age_minutes)
        return int(candidates[0]) if len(candidates) else None
    if strategy == "run_to_failure":
        return None
    if strategy == "rul_based":
        mask = group["predicted_rul_minutes"].to_numpy(float) <= policy.rul_warning_minutes
        return _persistent_first(mask, policy.consecutive_inspections)
    raise ValueError(f"Unknown strategy: {strategy}")


def replay_strategies(
    predictions: pd.DataFrame,
    policy: ReplayPolicy = ReplayPolicy(),
    costs: tuple[CostScenario, ...] = (CostScenario(),),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay three policies on one causal prediction trajectory per bearing."""
    # Ground-truth RUL is deliberately not required.  Decisions use only the
    # timestamp, the causal HI activation and the prediction available then.
    required = {"bearing_id", "condition", "step", "predicted_rul_minutes"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing replay columns: {sorted(missing)}")
    if predictions.duplicated(["bearing_id", "step"]).any():
        raise ValueError("Replay input must contain one row per bearing and step")

    rows: list[dict] = []
    for bearing, raw in predictions.groupby("bearing_id", sort=True):
        group = raw.sort_values("step").reset_index(drop=True)
        life = float(group["step"].max() * policy.inspection_cadence_minutes)
        for strategy in ("run_to_failure", "scheduled", "rul_based"):
            idx = _decision_row(group, strategy, policy)
            decision_age = float(group.loc[idx, "step"] * policy.inspection_cadence_minutes) if idx is not None else np.nan
            lead = life - decision_age if idx is not None else 0.0
            failed = idx is None
            late = bool(idx is not None and lead < policy.required_lead_minutes)
            early = bool(idx is not None and lead > policy.rul_warning_minutes)
            life_used = decision_age / life if idx is not None and life > 0 else 1.0
            for cost in costs:
                replacement_cost = cost.corrective_replacement if failed else cost.preventive_replacement
                downtime_cost = cost.failure_downtime if failed or late else 0.0
                # Every intervention before observed failure consumes unused
                # life, including a late intervention.  Apply this component
                # identically after the strategy-specific decision is made.
                premature_cost = (0.0 if failed else
                                  cost.premature_replacement * max(0.0, 1.0 - life_used))
                total = replacement_cost + downtime_cost + premature_cost
                rows.append({
                    "cost_scenario": cost.name, "strategy": strategy,
                    "bearing_id": bearing, "condition": group.loc[0, "condition"],
                    "observed_life_minutes": life, "decision_age_minutes": decision_age,
                    "lead_minutes": lead, "preventive_replacement": not failed,
                    "failure_before_intervention": failed, "early_replacement": early,
                    "late_intervention": late, "missed_warning": failed,
                    "life_fraction_used": life_used,
                    "life_consumed_percent": 100.0 * life_used,
                    "replacement_cost_units": replacement_cost,
                    "downtime_cost_units": downtime_cost,
                    "premature_penalty_units": premature_cost,
                    "simulated_cost_units": total,
                })
    events = pd.DataFrame(rows)
    summary = (events.groupby(["cost_scenario", "strategy"], as_index=False)
               .agg(bearings=("bearing_id", "nunique"),
                    preventive_replacements=("preventive_replacement", "sum"),
                    failures_before_intervention=("failure_before_intervention", "sum"),
                    early_replacements=("early_replacement", "sum"),
                    late_interventions=("late_intervention", "sum"),
                    missed_warnings=("missed_warning", "sum"),
                    mean_useful_lead_minutes=("lead_minutes", "mean"),
                    mean_life_fraction_used=("life_fraction_used", "mean"),
                    total_simulated_cost_units=("simulated_cost_units", "sum")))
    summary.attrs["policy"] = asdict(policy)
    return events, summary
