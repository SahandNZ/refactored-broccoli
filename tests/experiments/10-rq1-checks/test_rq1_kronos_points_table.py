"""The generator of the Kronos-points table (10-rq1-checks/make_tab_rq1_kronos_points.py)."""

import runpy
from collections.abc import Callable
from typing import Any, NamedTuple

import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import MARKET_NAMES, dashed_rule

SCRIPT = LAB_DIR / "experiments" / "10-rq1-checks" / "make_tab_rq1_kronos_points.py"


class Table(NamedTuple):
    """The functions of the script under test, typed."""

    format_skill: Callable[[float], str]
    body: Callable[..., str]
    values: Callable[..., str]


@pytest.fixture(name="table", scope="module")
def _table() -> Table:
    namespace: dict[str, Any] = runpy.run_path(str(SCRIPT), run_name="make_tab")
    return Table(
        namespace["format_skill"],
        namespace["kronos_points_body"],
        namespace["kronos_points_values"],
    )


def _verdict(skills: dict[str, float], *, passing: str = "") -> pd.DataFrame:
    # one scored cell per model and market at 1 bar, with the model's skill in every market
    rows = [
        {
            "model": model,
            "market": market,
            "horizon": 1,
            "status": "scored",
            "skill": skill,
            "beats_baseline": model == passing,
        }
        for model, skill in skills.items()
        for market in MARKET_NAMES
    ]
    return pd.DataFrame(rows)


def test_a_value_that_rounds_to_zero_prints_zero(table: Table):
    assert table.format_skill(-0.04) == "0.0"
    assert table.format_skill(-1.25) == "$-$1.2"


def test_rows_follow_the_paper_order_with_both_columns_per_market(table: Table):
    points = _verdict(
        {"kronos-small": -20.0, "chronos-bolt-small": -5.0}, passing="chronos-bolt-small"
    )
    rq1 = _verdict({"kronos-small": -20.0, "chronos-bolt-small": -3.0})
    lines = [line for line in table.body(points, rq1).splitlines() if not line.startswith("%")]
    assert lines[0].startswith("Selected baseline & 0.0")
    assert lines[2].startswith("Chronos-Bolt (S) & \\textbf{$-$5.0} & $-$3.0")
    assert lines[3] == dashed_rule(1 + 2 * len(MARKET_NAMES))
    assert lines[4].startswith("Kronos (S)")
    assert lines[2].count("&") == 2 * len(MARKET_NAMES)


def test_the_values_count_the_gaps_and_the_closure_shares(table: Table):
    points = _verdict({"kronos-small": -20.0, "kronos-base": -10.0, "tirex-1.1": -12.0})
    points.loc[points["market"] == "crypto", "skill"] = -1.0
    reopening = pd.DataFrame(
        {
            "market": list(MARKET_NAMES),
            "reopening_share": [0.0, 0.01, 0.02, 0.03, 0.14],
            "grid_reopening_share": [0.0, 0.33, 0.43, 0.46, 0.74],
            "target_closure_share": [0.0, 0.01, 0.03, 0.02, 0.13],
            "grid_target_closure_share": [0.0, 0.04, 0.05, 0.036, 0.045],
        }
    )
    rq1 = _verdict({"tirex-1.1": -1.0}).assign(times=8760)
    gaps = pd.DataFrame(
        {"gain": [5.0, 12.0, 20.0], "gain_low": [1.0, -1.0, float("nan")], "gain_high": 3 * [30.0]}
    )
    values = table.values(
        points.assign(times=55), reopening, gaps, rq1=rq1, options={"min_closure_hours": 12}
    )
    assert r"\newcommand{\kpGapAbove}{1}" in values
    assert r"\newcommand{\kpAcrossLow}{3.6}" in values
    assert r"\newcommand{\kpHourlyTimes}{8,760}" in values
    assert r"\newcommand{\kpGapPairs}{2}" in values
    assert r"\newcommand{\kpTimes}{55}" in values
    assert r"\newcommand{\kpReopenLow}{33.0}" in values
    assert r"\newcommand{\kpAllReopenHigh}{14.0}" in values
