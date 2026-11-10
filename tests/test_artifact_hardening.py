"""Regression test suite for artifact hardening and defensive error handling (Issue #5)."""

from pathlib import Path
import pytest
import pandas as pd

from bearing_rul.artifact_loader import (
    safe_load_csv,
    safe_load_json,
    ArtifactLoadResult,
)


@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Creates a temporary run artifact directory."""
    run_dir = tmp_path / "outputs" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_missing_file_handling(temp_run_dir: Path):
    """Verify handling of missing artifact files."""
    missing_file = temp_run_dir / "non_existent.csv"
    result = safe_load_csv(missing_file, run_name="test_run")

    assert not result.is_valid
    assert result.data is None
    assert result.warning_type == "MISSING_FILE"
    assert "test_run" in result.get_actionable_message()
    assert str(missing_file) in result.get_actionable_message()


def test_malformed_csv_handling(temp_run_dir: Path):
    """Verify handling of syntactically malformed / corrupted CSV files."""
    corrupt_file = temp_run_dir / "corrupt.csv"
    corrupt_file.write_bytes(b"\x00\xFF\xFE Garbage Binary Content \x80\x90")

    result = safe_load_csv(corrupt_file, run_name="test_run")

    assert not result.is_valid
    assert result.data is None
    assert result.warning_type in ["MALFORMED_CSV", "EMPTY_FILE"]
    assert "test_run" in result.get_actionable_message()


def test_missing_required_columns(temp_run_dir: Path):
    """Verify detection of CSV files missing required schema columns."""
    schema_file = temp_run_dir / "incomplete_schema.csv"
    df = pd.DataFrame({"bearing_id": ["Bearing1_1"], "wrong_column": [10.5]})
    df.to_csv(schema_file, index=False)

    result = safe_load_csv(
        schema_file,
        run_name="test_run",
        required_columns=["bearing_id", "actual_rul", "predicted_rul"],
    )

    assert not result.is_valid
    assert result.data is None
    assert result.warning_type == "MISSING_COLUMNS"
    assert "Missing required column(s)" in result.error_message


def test_non_numeric_values_in_numeric_columns(temp_run_dir: Path):
    """Verify detection of non-numeric data in numeric metric fields."""
    corrupt_values_file = temp_run_dir / "bad_types.csv"
    df = pd.DataFrame(
        {
            "bearing_id": ["Bearing1_1", "Bearing1_2"],
            "actual_rul": [120.0, "CORRUPTED_STRING_VALUE"],
            "predicted_rul": [115.5, 90.0],
        }
    )
    df.to_csv(corrupt_values_file, index=False)

    result = safe_load_csv(
        corrupt_values_file,
        run_name="test_run",
        required_columns=["bearing_id", "actual_rul", "predicted_rul"],
        numeric_columns=["actual_rul", "predicted_rul"],
    )

    assert not result.is_valid
    assert result.data is None
    assert result.warning_type == "NON_NUMERIC_VALUES"
    assert "Non-numeric values detected" in result.error_message


def test_valid_artifact_behavior_preserved(temp_run_dir: Path):
    """Verify that perfectly valid artifacts continue to load seamlessly."""
    valid_file = temp_run_dir / "valid_predictions.csv"
    df_valid = pd.DataFrame(
        {
            "bearing_id": ["Bearing1_1", "Bearing1_2"],
            "actual_rul": [100.0, 50.0],
            "predicted_rul": [98.5, 51.2],
        }
    )
    df_valid.to_csv(valid_file, index=False)

    result = safe_load_csv(
        valid_file,
        run_name="test_run",
        required_columns=["bearing_id", "actual_rul", "predicted_rul"],
        numeric_columns=["actual_rul", "predicted_rul"],
    )

    assert result.is_valid
    assert result.data is not None
    assert len(result.data) == 2
    assert result.data["predicted_rul"].iloc[0] == 98.5


def test_no_fabricated_results_on_failure(temp_run_dir: Path):
    """Verify that failure returns explicit None instead of fallback fabricated numbers."""
    empty_file = temp_run_dir / "empty.csv"
    empty_file.write_text("bearing_id,actual_rul,predicted_rul\n")

    result = safe_load_csv(
        empty_file,
        run_name="test_run",
        required_columns=["bearing_id", "actual_rul", "predicted_rul"],
    )

    assert not result.is_valid
    assert result.data is None