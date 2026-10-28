"""Standard RUL regression metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE and RMSE (in the same unit as the target)."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
