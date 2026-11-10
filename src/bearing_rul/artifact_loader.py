from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import pandas as pd


@dataclass
class ArtifactLoadResult:
    """Encapsulates the result of loading an experiment artifact with diagnostic metadata."""

    data: Optional[Any]
    is_valid: bool
    run_name: str
    source_path: Path
    error_message: Optional[str] = None
    warning_type: Optional[str] = None

    def get_actionable_message(self) -> str:
        """Returns a user-facing, actionable diagnostic message."""
        if self.is_valid:
            return f"Successfully loaded artifact from '{self.source_path}' (Run: '{self.run_name}')."
        return (
            f"[{self.warning_type or 'ARTIFACT_ERROR'}] Run: '{self.run_name}' | "
            f"Source: '{self.source_path}' | Details: {self.error_message}"
        )


def safe_load_csv(
    file_path: Path,
    run_name: str,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
) -> ArtifactLoadResult:
    """Safely loads a CSV artifact with defensive validation.

    Guarantees no uncaught exceptions and prevents returning fabricated or corrupted data.
    """
    path_obj = Path(file_path)

    # 1. Missing File Check
    if not path_obj.exists():
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message="File does not exist on disk.",
            warning_type="MISSING_FILE",
        )

    # 2. Syntax & Empty Data Parsing Check
    try:
        df = pd.read_csv(path_obj)
    except pd.errors.EmptyDataError:
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message="File is empty or contains no readable data.",
            warning_type="EMPTY_FILE",
        )
    except Exception as exc:
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message=f"Malformed CSV syntax or unreadable format: {str(exc)}",
            warning_type="MALFORMED_CSV",
        )

    if df.empty:
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message="CSV contains header column definitions but zero data rows.",
            warning_type="EMPTY_DATASET",
        )

    # 3. Column Schema Check
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return ArtifactLoadResult(
                data=None,
                is_valid=False,
                run_name=run_name,
                source_path=path_obj,
                error_message=f"Missing required column(s): {missing_cols}.",
                warning_type="MISSING_COLUMNS",
            )

    # 4. Non-numeric Data Check
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.isna().any():
                    invalid_rows = df[df[col].isna() | converted.isna()][col].tolist()
                    return ArtifactLoadResult(
                        data=None,
                        is_valid=False,
                        run_name=run_name,
                        source_path=path_obj,
                        error_message=(
                            f"Non-numeric values detected in numeric column '{col}': "
                            f"{invalid_rows[:3]}."
                        ),
                        warning_type="NON_NUMERIC_VALUES",
                    )

    return ArtifactLoadResult(
        data=df,
        is_valid=True,
        run_name=run_name,
        source_path=path_obj,
    )


def safe_load_json(file_path: Path, run_name: str) -> ArtifactLoadResult:
    """Safely loads a JSON summary artifact with defensive validation."""
    path_obj = Path(file_path)

    if not path_obj.exists():
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message="File does not exist on disk.",
            warning_type="MISSING_FILE",
        )

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ArtifactLoadResult(
            data=data,
            is_valid=True,
            run_name=run_name,
            source_path=path_obj,
        )
    except Exception as exc:
        return ArtifactLoadResult(
            data=None,
            is_valid=False,
            run_name=run_name,
            source_path=path_obj,
            error_message=f"JSON decoding error or corrupted structure: {str(exc)}",
            warning_type="MALFORMED_JSON",
        )