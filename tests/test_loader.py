from bearing_rul.data.xjtu_loader import list_measurements


def test_numeric_filename_order(tmp_path):
    bearing = tmp_path / "condition" / "Bearing1_1"
    bearing.mkdir(parents=True)
    for name in ("10.csv", "1.csv", "2.csv"):
        (bearing / name).write_text("1,2\n3,4\n", encoding="utf-8")
    result = list_measurements(tmp_path)
    assert result["filename"].tolist() == ["1.csv", "2.csv", "10.csv"]
    assert result["step"].tolist() == [0, 1, 2]
