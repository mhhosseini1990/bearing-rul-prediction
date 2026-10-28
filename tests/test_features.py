import numpy as np
import pytest
from bearing_rul.features.time_domain import extract_features, extract_features_from_csv


def test_constant_signal():
    features = extract_features(np.ones(4), np.zeros(4))
    assert features["horizontal_rms"] == pytest.approx(1.0)
    assert features["horizontal_std"] == pytest.approx(0.0)
    assert features["vertical_crest_factor"] == pytest.approx(0.0)


def test_csv_with_header(tmp_path):
    sample = tmp_path / "1.csv"
    sample.write_text("horizontal,vertical\n1,2\n3,4\n", encoding="utf-8")
    features = extract_features_from_csv(sample)
    assert features["horizontal_mean"] == pytest.approx(2.0)
    assert features["vertical_mean"] == pytest.approx(3.0)
