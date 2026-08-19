from pathlib import Path

import pandas as pd
import pytest

from analysis import load_csv_results, rank_results, summarize_results, validate_result_table


ROOT = Path(__file__).resolve().parents[1]


def test_load_committed_results():
    results = load_csv_results(ROOT / "results")
    assert not results.empty
    assert {"Metric", "K", "Accuracy", "Source"}.issubset(results.columns)
    assert results["Accuracy"].between(0, 1).all()


def test_rank_results_orders_accuracy_descending():
    frame = pd.DataFrame(
        {"Metric": ["manhattan", "euclidean"], "K": [3, 1], "Accuracy": [0.91, 0.97]}
    )
    ranked = rank_results(frame)
    assert ranked.iloc[0]["Metric"] == "euclidean"
    assert ranked.iloc[0]["Accuracy"] == pytest.approx(0.97)


def test_summary_reports_best_configuration():
    frame = pd.DataFrame(
        {"Metric": ["manhattan", "euclidean"], "K": [3, 1], "Accuracy": [0.91, 0.97]}
    )
    summary = summarize_results(frame)
    assert summary["rows"] == 2
    assert summary["best_metric"] == "euclidean"
    assert summary["best_k"] == 1
    assert summary["best_accuracy"] == pytest.approx(0.97)


def test_invalid_accuracy_is_rejected():
    frame = pd.DataFrame({"Metric": ["euclidean"], "K": [1], "Accuracy": [1.2]})
    with pytest.raises(ValueError, match="outside"):
        validate_result_table(frame)
