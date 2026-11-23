"""The generator of the paper's ranking table (01-universe/make_tab_universe_ranking.py)."""

import runpy
from collections.abc import Callable
from types import ModuleType
from typing import Any, NamedTuple

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import MARKET_NAMES
from src.experiment import load_study

STUDY = LAB_DIR / "experiments" / "01-universe"
SCRIPT = STUDY / "make_tab_universe_ranking.py"


class Table(NamedTuple):
    """The functions of the script under test, typed."""

    tau: Callable[[float], str]
    body: Callable[..., str]
    values: Callable[..., str]


@pytest.fixture(name="table")
def _table(study_lib_as_lib: ModuleType) -> Table:
    del study_lib_as_lib
    namespace: dict[str, Any] = runpy.run_path(str(SCRIPT), run_name="make_tab")
    return Table(namespace["_tau"], namespace["_body"], namespace["_values"])


@pytest.fixture(name="settings")
def _settings(study_lib_as_lib: ModuleType) -> Any:
    return study_lib_as_lib.read_ranking_settings(load_study(STUDY / "study.toml").options)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # two groups, every market and quarter, 20 splits; markets 0.5 in the grid group
    splits = pd.concat(
        [
            pd.DataFrame(
                {
                    "group": group,
                    "split": range(20),
                    "tau": tau,
                    "same_top": [True] * 15 + [False] * 5,
                }
            )
            for group, tau in (("hourly", 0.9), ("grid", 0.6))
        ]
    )
    parts = pd.DataFrame(
        [
            {"group": group, "check": "market", "part": market, "tau": tau}
            for group, tau in (("hourly", 0.8), ("grid", 0.5))
            for market in MARKET_NAMES
        ]
        + [
            {"group": group, "check": "quarter", "part": window, "tau": -0.004}
            for group in ("hourly", "grid")
            for window in ("t1", "t2", "t3", "t4")
        ]
    )
    scores = pd.DataFrame(
        {
            "group": ["hourly"] * 3 + ["grid"] * 4,
            "model": ["a", "b", "c", "a", "b", "c", "k"],
            "kind": ["baseline", "zero-shot", "zero-shot", "baseline", "zero-shot", "zero-shot",
                     "zero-shot"],
            "decisions": [100] * 3 + [10] * 4,
        }
    )  # fmt: skip
    return splits, parts, scores


def test_the_real_study_toml_has_the_ranking_settings(settings: Any):
    assert settings.windows == ("t1", "t2", "t3", "t4")
    assert 0 < settings.range_level < 1


def test_a_tau_that_rounds_to_zero_prints_zero(table: Table):
    assert table.tau(-0.004) == "0.00"
    assert table.tau(-0.25) == "$-$0.25"
    assert table.tau(0.256) == "0.26"


def test_the_body_lists_markets_in_paper_order_then_quarters(table: Table, settings: Any):
    splits, parts, _ = _tables()
    body = table.body(splits, parts, settings)
    names = [line.split(" vs.")[0] for line in body.splitlines() if " vs." in line]
    assert names == [*MARKET_NAMES.values(), "t1", "t2", "t3", "t4"]
    assert "& 75.0 & 75.0" in body


def test_the_values_count_the_markets_below_the_threshold(table: Table, settings: Any):
    splits, parts, scores = _tables()
    values = table.values(splits, parts, scores, settings)
    assert r"\newcommand{\rankGridMarketsBelow}{5}" in values
    assert r"\newcommand{\rankBaselines}{1}" in values
    assert r"\newcommand{\rankPairs}{3}" in values
