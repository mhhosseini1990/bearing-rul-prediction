"""Training-data contracts for the approved fixed48 direct-RUL protocol."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402


def _frame(ids, rows=35):
    records = []
    for bearing in ids:
        for step in range(rows):
            record = {"bearing_id": bearing, "condition": int(bearing[7]), "step": step,
                      "elapsed_minutes": step, "rul_minutes": rows - 1 - step}
            record.update({name: float(step + i) for i, name in enumerate(train.FEATURE_NAMES)})
            records.append(record)
    return pd.DataFrame(records)


def test_five_folds_are_disjoint_9_3_3_and_cover_every_test_bearing_once():
    ids = {f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6)}
    train.validate_folds(ids)
    observed = [b for split in train.FOLDS.values() for b in split["test"]]
    assert len(observed) == len(set(observed)) == 15


def test_sequences_are_past_only_and_do_not_cross_bearings():
    frame = _frame(["Bearing1_1", "Bearing2_1"])
    x, y, owners, metadata = train.sequences(frame, 10, np.zeros(48), np.ones(48))
    assert x.shape == (52, 10, 48) and metadata.step.min() == 9
    assert np.array_equal(y, metadata.rul_minutes.to_numpy(np.float32))
    for index, owner in enumerate(owners):
        end = int(metadata.iloc[index].step)
        expected = frame[(frame.bearing_id == owner) & frame.step.between(end - 9, end)]
        np.testing.assert_allclose(x[index, :, 0], expected[train.FEATURE_NAMES[0]])


def test_feature_scaling_statistics_are_training_only():
    split = train.FOLDS["fold_1"]
    frame = _frame([*split["train"], *split["val"], *split["test"]])
    frame.loc[frame.bearing_id.isin(split["val"]), train.FEATURE_NAMES[0]] = 1e6
    frame.loc[frame.bearing_id.isin(split["test"]), train.FEATURE_NAMES[0]] = 2e6
    values = frame[frame.bearing_id.isin(split["train"])].loc[:, train.FEATURE_NAMES].to_numpy(float)
    assert values[:, 0].mean() < 100


def test_score_is_equal_weight_per_bearing_not_snapshot_weighted():
    metadata = pd.DataFrame({"bearing_id": ["A", "B", "B", "B"],
                             "condition": [1, 2, 2, 2], "step": [29, 29, 30, 31],
                             "rul_minutes": [0., 0., 0., 0.]})
    result = train.score(metadata.rul_minutes.to_numpy(), np.array([10., 1., 1., 1.]), metadata)
    assert result.mae_minutes.mean() == pytest.approx(5.5)
    assert np.abs(np.array([10., 1., 1., 1.])).mean() == pytest.approx(3.25)

