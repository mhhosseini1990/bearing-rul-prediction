import pandas as pd
from bearing_rul.data.splits import split_by_bearing


def test_bearings_are_not_shared():
    df = pd.DataFrame({"bearing_id": ["a", "a", "b", "b", "c", "c", "d", "d"], "step": [0, 1] * 4})
    train, test = split_by_bearing(df, test_size=0.25)
    assert len(train) > 0 and len(test) > 0
    assert set(train["bearing_id"]).isdisjoint(set(test["bearing_id"]))
