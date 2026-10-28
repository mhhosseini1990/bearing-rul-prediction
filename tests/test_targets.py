import pandas as pd
from bearing_rul.data.rul_labels import add_remaining_recordings


def test_remaining_recordings_per_bearing():
    df = pd.DataFrame({"bearing_id": ["a", "a", "a", "b", "b"], "step": [0, 1, 2, 0, 1]})
    labelled = add_remaining_recordings(df)
    assert labelled["rul_remaining_recordings"].tolist() == [2, 1, 0, 1, 0]
