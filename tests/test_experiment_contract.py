from pathlib import Path

import pandas as pd

from fashion_gridsearch.config import number_of_candidates, number_of_fits


def test_search_space_matches_executed_run() -> None:
    assert number_of_candidates() == 12
    assert number_of_fits() == 36


def test_captured_run_is_complete() -> None:
    summary = pd.read_csv("results/run_summary.csv").iloc[0]
    assert summary["total_fits"] == 36
    assert summary["mean_cv_accuracy"] == 0.8841
    assert summary["test_accuracy"] == 0.8748


def test_notebook_has_no_unfilled_conclusion_placeholders() -> None:
    notebook = Path("notebooks/fashion_mnist_gridsearch.ipynb").read_text(encoding="utf-8")
    assert "Completar después de ejecutar" not in notebook

