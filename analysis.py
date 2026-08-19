"""Reusable helpers for the MNIST and letter-recognition KNN artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"Metric", "K", "Accuracy"}


def _header_row(path: Path) -> int:
    """Find the first CSV row containing the experiment result header."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            columns = {part.strip() for part in line.rstrip("\n").split(",")}
            if REQUIRED_COLUMNS.issubset(columns):
                return index
    raise ValueError(f"No KNN result header found in {path.name}")


def validate_result_table(frame: pd.DataFrame, *, source: str = "results") -> pd.DataFrame:
    """Validate and normalize a KNN result table without changing its source file."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    normalized = frame.copy()
    normalized["Metric"] = normalized["Metric"].astype(str).str.strip().str.lower()
    normalized["K"] = pd.to_numeric(normalized["K"], errors="coerce")
    normalized["Accuracy"] = pd.to_numeric(normalized["Accuracy"], errors="coerce")
    normalized = normalized.dropna(subset=["Metric", "K", "Accuracy"])
    normalized = normalized[normalized["Metric"].ne("")]
    if normalized.empty:
        raise ValueError(f"{source} contains no valid KNN result rows")
    if normalized["K"].le(0).any():
        raise ValueError(f"{source} contains non-positive K values")
    if normalized["Accuracy"].lt(0).any() or normalized["Accuracy"].gt(1).any():
        raise ValueError(f"{source} contains accuracy values outside [0, 1]")
    return normalized.reset_index(drop=True)


def load_csv_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all compatible CSV result exports from a results directory."""
    directory = Path(results_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        try:
            frame = pd.read_csv(path, skiprows=_header_row(path))
            frame = validate_result_table(frame, source=path.name)
        except (OSError, pd.errors.ParserError, ValueError):
            continue
        frame.insert(0, "Source", path.name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No compatible KNN CSV results found in {directory}")
    return pd.concat(frames, ignore_index=True)


def rank_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return experiment rows sorted from highest to lowest accuracy."""
    validated = validate_result_table(frame)
    return validated.sort_values(["Accuracy", "K"], ascending=[False, True]).reset_index(drop=True)


def summarize_results(frame: pd.DataFrame) -> dict[str, object]:
    """Create dashboard KPI values from validated experiment rows."""
    ranked = rank_results(frame)
    best = ranked.iloc[0]
    return {
        "rows": int(len(ranked)),
        "metrics": sorted(ranked["Metric"].unique().tolist()),
        "best_accuracy": float(best["Accuracy"]),
        "best_metric": str(best["Metric"]),
        "best_k": int(best["K"]),
    }


def available_metrics(frame: pd.DataFrame) -> Iterable[str]:
    """Return stable alphabetical metric options for a filter control."""
    return sorted(validate_result_table(frame)["Metric"].unique().tolist())
