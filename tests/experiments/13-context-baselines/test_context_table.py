"""The macros of the context-baseline check (make_tab_rq1_context_baselines.py)."""

import runpy
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src import LAB_DIR

SCRIPT = (
    LAB_DIR / "experiments" / "13-context-baselines"
    / "make_tab_rq1_context_baselines.py"
)  # fmt: skip


@pytest.fixture(name="values", scope="module")
def _values() -> Callable[..., str]:
    namespace: dict[str, Any] = runpy.run_path(str(SCRIPT), run_name="make_tab")
    return namespace["context_values"]


def _tables(baselines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verdict = pd.DataFrame(
        {"status": ["scored", "scored", "untested"], "beats_baseline": [False, True, False]}
    )
    compared = pd.DataFrame(
        {"market": ["crypto", "forex"], "baseline": baselines, "skill": [-9.0, -9.2],
         "skill_rq1": [-9.3, -9.1]}
    )  # fmt: skip
    failed = pd.DataFrame({"asset": ["A", "B"], "failed": [3, 1]})
    return verdict, compared, failed


def test_the_macros_count_the_cells_and_name_the_best_baseline(values: Callable[..., str]):
    text = values(*_tables(["climatology", "climatology"]))
    assert r"\newcommand{\cbTested}{2}" in text
    assert r"\newcommand{\cbPassing}{1}" in text
    assert r"\newcommand{\cbBest}{climatology}" in text
    assert r"\newcommand{\cbGarchFailed}{4}" in text
    assert r"\newcommand{\cbMedian}{$-$9.1}" in text


def test_different_best_baselines_raise(values: Callable[..., str]):
    with pytest.raises(ValueError, match="differ"):
        values(*_tables(["climatology", "linear"]))
