import pandas as pd
import pytest

from bearing_rul.maintenance import CostScenario, ReplayPolicy, replay_strategies


def test_three_strategies_are_causal_and_counted_once_per_bearing():
    data = pd.DataFrame([{"bearing_id": "B1", "condition": "C1", "step": step,
                          "rul_minutes": 10-step, "predicted_rul_minutes": 10-step}
                         for step in range(11)])
    events, summary = replay_strategies(data, ReplayPolicy(8, 4, 2, 2), (CostScenario(),))
    assert set(events.strategy) == {"run_to_failure", "scheduled", "rul_based"}
    assert len(events) == 3 and summary.bearings.eq(1).all()
    assert events.loc[events.strategy == "run_to_failure", "failure_before_intervention"].iat[0]


def test_duplicate_bearing_steps_are_rejected():
    row = {"bearing_id": "B", "condition": "C", "step": 0, "rul_minutes": 0,
           "predicted_rul_minutes": 0}
    with pytest.raises(ValueError, match="one row"):
        replay_strategies(pd.DataFrame([row, row]))


def test_ground_truth_rul_is_not_required_for_decisions():
    data = pd.DataFrame([{"bearing_id": "B", "condition": "C", "step": step,
                          "predicted_rul_minutes": 4-step} for step in range(5)])
    events, _ = replay_strategies(data, ReplayPolicy(3, 2, 1, 2), (CostScenario(),))
    assert set(events.dropna(subset=["decision_age_minutes"]).decision_age_minutes) == {3.0}


def test_late_preventive_action_keeps_premature_penalty_component():
    data = pd.DataFrame([{"bearing_id": "B", "condition": "C", "step": step,
                          "predicted_rul_minutes": 10-step} for step in range(11)])
    events, _ = replay_strategies(data, ReplayPolicy(9, 2, 3, 2), (CostScenario(),))
    late = events[events.strategy == "scheduled"].iloc[0]
    assert late.late_intervention
    assert late.premature_penalty_units == pytest.approx(0.1)
    assert late.simulated_cost_units == pytest.approx(
        late.replacement_cost_units + late.downtime_cost_units + late.premature_penalty_units)
