"""Historical maintenance replay using *saved* RUL predictions.

This is a decision-rule simulation, not industrial validation. The alert policy
uses predicted RUL from current/past rows only. True RUL is used afterward to
assess lead time; it is never used to trigger an alert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "bearing_id": ("bearing_id", "bearing", "bearing_name", "test_bearing", "unit_id"),
    "step": ("target_step", "step", "snapshot_index", "snapshot_step", "time_step", "sequence_end_step", "file_index", "index_in_bearing"),
    "actual_rul_minutes": (
        "actual_rul_minutes", "true_rul_minutes", "rul_true_minutes",
        "rul_actual_minutes", "y_true_minutes", "actual_rul",
        "true_rul", "rul_true", "rul_minutes", "y_true",
    ),
    "predicted_rul_minutes": (
        "predicted_rul_minutes", "prediction_minutes", "pred_rul_minutes",
        "rul_pred_minutes", "y_pred_minutes", "predicted_rul",
        "prediction", "pred_rul", "rul_pred", "y_pred",
    ),
}


@dataclass(frozen=True)
class ReplaySettings:
    threshold_minutes: float = 120.0
    consecutive_snapshots: int = 3
    minimum_lead_minutes: float = 30.0

    def __post_init__(self) -> None:
        if self.threshold_minutes <= 0:
            raise ValueError("threshold_minutes must be > 0")
        if self.consecutive_snapshots < 1:
            raise ValueError("consecutive_snapshots must be >= 1")
        if self.minimum_lead_minutes < 0 or self.minimum_lead_minutes > self.threshold_minutes:
            raise ValueError("minimum_lead_minutes must be in [0, threshold_minutes]")


def _select_column(frame: pd.DataFrame, canonical: str) -> str:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    matched = [lookup[name] for name in COLUMN_ALIASES[canonical] if name in lookup]
    if not matched:
        raise ValueError(
            f"Missing {canonical!r} column. Supported names: "
            f"{', '.join(COLUMN_ALIASES[canonical])}. Found: {list(frame.columns)}"
        )
    if len(matched) > 1:
        raise ValueError(f"Ambiguous columns for {canonical}: {matched}")
    return matched[0]


def normalize_predictions(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Validate schema and chronological consistency without inventing step order."""
    if frame.empty:
        raise ValueError(f"{model}: prediction CSV is empty")
    if not model.strip():
        raise ValueError("model name cannot be empty")
    renames = { _select_column(frame, key): key for key in COLUMN_ALIASES }
    normalized = frame.rename(columns=renames)[list(COLUMN_ALIASES)].copy()
    normalized.insert(0, "model", model.upper())
    normalized["bearing_id"] = normalized["bearing_id"].astype("string").str.strip()
    if normalized["bearing_id"].isna().any() or normalized["bearing_id"].eq("").any():
        raise ValueError(f"{model}: blank bearing_id")
    for key in ("step", "actual_rul_minutes", "predicted_rul_minutes"):
        normalized[key] = pd.to_numeric(normalized[key], errors="coerce")
        if normalized[key].isna().any() or not normalized[key].map(lambda x: abs(x) != float("inf")).all():
            raise ValueError(f"{model}: {key} must contain finite numeric values")
    if (normalized["actual_rul_minutes"] < 0).any():
        raise ValueError(f"{model}: actual RUL cannot be negative")
    # Predictions may have been clipped to 0 during evaluation; clamp for rule safety.
    normalized["predicted_rul_minutes"] = normalized["predicted_rul_minutes"].clip(lower=0.0)
    if normalized.duplicated(["bearing_id", "step"]).any():
        raise ValueError(f"{model}: duplicate bearing_id + step rows")
    normalized = normalized.sort_values(["bearing_id", "step"], kind="stable").reset_index(drop=True)
    for bearing, rows in normalized.groupby("bearing_id", sort=False):
        if (rows["step"].diff().dropna() <= 0).any():
            raise ValueError(f"{model}/{bearing}: step order is not strictly increasing")
        if (rows["actual_rul_minutes"].diff().dropna() > 0.000001).any():
            raise ValueError(f"{model}/{bearing}: actual RUL increases with time; check labels/order")
        if rows["actual_rul_minutes"].iloc[-1] > 0.000001:
            raise ValueError(
                f"{model}/{bearing}: last actual RUL is not zero; "
                "cannot claim a full run-to-failure replay on an incomplete history"
            )
    return normalized


def first_alert_index(predicted_rul: Iterable[float], settings: ReplaySettings) -> int | None:
    """Return first online alert index using only predictions seen up to that point."""
    count = 0
    for index, value in enumerate(predicted_rul):
        count = count + 1 if float(value) <= settings.threshold_minutes else 0
        if count >= settings.consecutive_snapshots:
            return index
    return None


def replay_bearing(rows: pd.DataFrame, settings: ReplaySettings) -> tuple[dict, pd.DataFrame]:
    """Replay one already normalized model/bearing series, returning event and trace."""
    if rows.empty or rows["model"].nunique() != 1 or rows["bearing_id"].nunique() != 1:
        raise ValueError("Expected one nonempty model/bearing series")
    rows = rows.sort_values("step", kind="stable").reset_index(drop=True).copy()
    idx = first_alert_index(rows["predicted_rul_minutes"].tolist(), settings)
    rows["below_threshold"] = rows["predicted_rul_minutes"] <= settings.threshold_minutes
    rows["alert_issued"] = False
    rows["alert_active"] = False
    if idx is not None:
        rows.loc[idx, "alert_issued"] = True
        rows.loc[idx:, "alert_active"] = True
    if idx is None:
        status, lead, step, predicted = "missed", None, None, None
    else:
        step = float(rows.loc[idx, "step"])
        lead = float(rows.loc[idx, "actual_rul_minutes"])
        predicted = float(rows.loc[idx, "predicted_rul_minutes"])
        if lead > settings.threshold_minutes:
            status = "early"
        elif lead < settings.minimum_lead_minutes:
            status = "late"
        else:
            status = "timely"
    event = {
        "model": str(rows.loc[0, "model"]),
        "bearing_id": str(rows.loc[0, "bearing_id"]),
        "n_predictions": int(len(rows)),
        "first_step": float(rows.loc[0, "step"]),
        "last_step": float(rows.loc[len(rows)-1, "step"]),
        "alert_step": step,
        "predicted_rul_at_alert_minutes": predicted,
        "actual_lead_time_minutes": lead,
        "status": status,
    }
    return event, rows


def run_replay(predictions: pd.DataFrame, settings: ReplaySettings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return event-level records, model-level summary, and step-by-step traces."""
    events: list[dict] = []
    traces: list[pd.DataFrame] = []
    for (_model, _bearing), rows in predictions.groupby(["model", "bearing_id"], sort=True):
        event, trace = replay_bearing(rows, settings)
        events.append(event)
        traces.append(trace)
    events_df = pd.DataFrame.from_records(events)
    trace_df = pd.concat(traces, ignore_index=True)
    summaries = []
    for model, rows in events_df.groupby("model", sort=True):
        leads = rows["actual_lead_time_minutes"].dropna()
        counts = rows["status"].value_counts()
        summaries.append({
            "model": model,
            "bearings": len(rows),
            "timely": int(counts.get("timely", 0)),
            "early": int(counts.get("early", 0)),
            "late": int(counts.get("late", 0)),
            "missed": int(counts.get("missed", 0)),
            "timely_rate_pct": round(100 * int(counts.get("timely", 0)) / len(rows), 2),
            "mean_alert_lead_minutes": round(float(leads.mean()), 3) if not leads.empty else None,
        })
    return events_df, pd.DataFrame(summaries), trace_df


def load_prediction_files(lstm_csv: str | Path, gru_csv: str | Path) -> pd.DataFrame:
    """Read predictions from two existing evaluation files; no model retraining."""
    lfile, gfile = Path(lstm_csv), Path(gru_csv)
    for path in (lfile, gfile):
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file missing: {path}")
    lstm = normalize_predictions(pd.read_csv(lfile), "LSTM")
    gru = normalize_predictions(pd.read_csv(gfile), "GRU")
    # Fair side-by-side comparison: exactly the same bearing/time/ground-truth pairs.
    keys = ["bearing_id", "step", "actual_rul_minutes"]
    left = lstm[keys].reset_index(drop=True)
    right = gru[keys].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("LSTM and GRU prediction CSVs differ in bearing, step, or actual RUL alignment")
    return pd.concat([lstm, gru], ignore_index=True)


def save_replay_results(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    trace: pd.DataFrame,
    settings: ReplaySettings,
    destination: str | Path,
) -> None:
    import json
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events.to_csv(destination / "replay_events.csv", index=False)
    summary.to_csv(destination / "replay_summary.csv", index=False)
    trace.to_csv(destination / "replay_timeline.csv", index=False)
    (destination / "replay_config.json").write_text(
        json.dumps({**asdict(settings), "purpose": "illustrative_offline_historical_replay",
                    "data_source": "saved_model_predictions", "not_validated_for_industrial_use": True},
                   indent=2), encoding="utf-8")
