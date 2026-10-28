"""Planned LSTM sequence model: intentionally NOT IMPLEMENTED yet.

Future work:
1. Build chronological input windows separately within each bearing.
2. Fit normalization on training bearings only.
3. Train on training bearings; tune on validation bearings.
4. Evaluate on held-out test bearings using the same RUL units as RF.
5. Make sure historical replay never sees future windows.
"""


def build_lstm_model(*args, **kwargs):
    raise NotImplementedError("LSTM model will be added in a later milestone.")
