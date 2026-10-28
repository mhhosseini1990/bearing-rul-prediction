import pytest
from bearing_rul.evaluation.metrics import regression_metrics


def test_mae_rmse():
    metrics = regression_metrics([0, 2], [1, 1])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
