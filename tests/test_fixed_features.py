"""Fixed 48-feature extraction, registry, and prepared-data tests."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from bearing_rul.features.fixed_features import FEATURE_GROUPS, FEATURE_NAMES, FEATURE_SET_SHA256, FEATURE_SET_VERSION, extract_fixed_features
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/processed/xjtu_fixed_features_48.csv"
MANIFEST = CSV.with_suffix(".manifest.json")


def test_registry_has_exact_groups_and_expected_hash():
    assert FEATURE_SET_VERSION == "fixed48_v1"
    assert [len(group) for group in FEATURE_GROUPS.values()] == [8, 16, 16, 8]
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 48
    assert FEATURE_SET_SHA256 == "18692d899aafd55be58a5a01fd66892c2b7f8cb8e1bdad75d2f09a34deaf2b0f"


def test_extraction_is_deterministic_finite_and_uses_both_channels():
    signal = np.random.default_rng(2026).normal(size=(1024, 2))
    first = extract_fixed_features(signal); second = extract_fixed_features(signal.copy())
    assert tuple(first) == FEATURE_NAMES
    np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=0, atol=0)
    changed = signal.copy(); changed[:, 1] *= 3
    third = extract_fixed_features(changed)
    assert first["horizontal_A3_energy"] == third["horizontal_A3_energy"]
    assert first["vertical_A3_energy"] != third["vertical_A3_energy"]


def test_dwt_and_wpt_features_are_nonnegative():
    result = extract_fixed_features(np.random.default_rng(1).normal(size=(1024, 2)))
    assert all(result[name] >= 0 for name in (*FEATURE_GROUPS["dwt"], *FEATURE_GROUPS["wpt"]))


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError): extract_fixed_features(np.zeros((1024, 1)))
    invalid = np.zeros((1024, 2)); invalid[0, 0] = np.nan
    with pytest.raises(ValueError): extract_fixed_features(invalid)


def test_canonical_csv_and_manifest_match_verified_contract():
    digest = hashlib.sha256(CSV.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frame = pd.read_csv(CSV)
    assert digest == manifest["sha256"] == "d72fb0c3d05badf455ccf9cc0aceac5a898d7d30b7c13e424feacc669420451d"
    assert manifest["feature_registry_sha256"] == FEATURE_SET_SHA256
    assert (len(frame), frame.bearing_id.nunique(), frame.condition.nunique()) == (9216, 15, 3)
    assert list(frame.columns[-48:]) == list(FEATURE_NAMES)
