"""Tests for the XJTU-SY vibration CSV loader."""

import numpy as np
import pandas as pd
import pytest

from bearing_rul.data.xjtu_loader import (
    list_bearing_csvs,
    load_vibration_csv,
    list_measurements,
)


EXPECTED_COLUMNS = [
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
]


def test_d01_load_valid_two_channel_csv(tmp_path):
    """D-01: valid CSV preserves rows, columns and numeric values."""

    path = tmp_path / "1.csv"

    pd.DataFrame(
        {
            EXPECTED_COLUMNS[0]: [0.5, -0.2, 1.25],
            EXPECTED_COLUMNS[1]: [-0.1, 0.3, 0.0],
        }
    ).to_csv(path, index=False)

    values = load_vibration_csv(path)

    assert isinstance(values, np.ndarray)

    assert values.shape == (3, 2)

    np.testing.assert_allclose(
        values,
        [
            [0.5, -0.1],
            [-0.2, 0.3],
            [1.25, 0.0],
        ],
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "empty",
        "bad_columns",
        "non_numeric",
        "nan",
    ],
)
def test_d02_reject_invalid_csv(tmp_path, kind):
    """D-02: invalid CSV inputs must raise explicit errors."""

    path = tmp_path / "1.csv"

    if kind == "empty":
        path.write_text(
            "",
            encoding="utf-8",
        )

    elif kind == "bad_columns":
        path.write_text(
            "wrong,columns\n1,2\n",
            encoding="utf-8",
        )

    elif kind == "non_numeric":
        path.write_text(
            ",".join(EXPECTED_COLUMNS) + "\nabc,1\n",
            encoding="utf-8",
        )

    elif kind == "nan":
        path.write_text(
            ",".join(EXPECTED_COLUMNS) + "\n,1\n",
            encoding="utf-8",
        )

    with pytest.raises(
        (FileNotFoundError, ValueError)
    ):
        load_vibration_csv(path)


def test_d03_numeric_chronological_order(tmp_path):
    """D-03: snapshots must be ordered numerically."""

    for filename in [
        "10.csv",
        "2.csv",
        "1.csv",
    ]:
        (tmp_path / filename).write_text(
            ",".join(EXPECTED_COLUMNS) + "\n1,2\n",
            encoding="utf-8",
        )

    # A gap is intentionally present, so complete the sequence.
    for number in range(3, 10):
        (tmp_path / f"{number}.csv").write_text(
            ",".join(EXPECTED_COLUMNS) + "\n1,2\n",
            encoding="utf-8",
        )

    files = list_bearing_csvs(tmp_path)

    assert [
        path.name for path in files
    ] == [
        f"{number}.csv"
        for number in range(1, 11)
    ]


def test_d03_reject_missing_snapshot_number(tmp_path):
    """D-03: missing snapshot numbers must raise an error."""

    for filename in [
        "1.csv",
        "3.csv",
    ]:
        (tmp_path / filename).write_text(
            ",".join(EXPECTED_COLUMNS) + "\n1,2\n",
            encoding="utf-8",
        )

    with pytest.raises(
        ValueError,
        match="Missing|sequence|consecutive",
    ):
        list_bearing_csvs(tmp_path)


def test_manifest_preserves_bearing_identity(tmp_path):
    """Manifest keeps operating conditions and bearings separate."""

    bearing_dir = (
        tmp_path
        / "35Hz12kN"
        / "Bearing1_1"
    )

    bearing_dir.mkdir(parents=True)

    for number in range(1, 4):
        (bearing_dir / f"{number}.csv").write_text(
            ",".join(EXPECTED_COLUMNS) + "\n1,2\n",
            encoding="utf-8",
        )

    manifest = list_measurements(tmp_path)

    assert len(manifest) == 3

    assert manifest["bearing_name"].unique().tolist() == [
        "Bearing1_1"
    ]

    assert manifest["operating_condition"].unique().tolist() == [
        "35Hz12kN"
    ]

    assert manifest["step"].tolist() == [
        0,
        1,
        2,
    ]

    assert manifest["filename"].tolist() == [
        "1.csv",
        "2.csv",
        "3.csv",
    ]