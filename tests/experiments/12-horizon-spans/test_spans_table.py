"""The macros of the horizon spans (12-horizon-spans/make_tab_protocol_horizon_spans.py)."""

import runpy
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src import LAB_DIR

SCRIPT = (
    LAB_DIR / "experiments" / "12-horizon-spans"
    / "make_tab_protocol_horizon_spans.py"
)  # fmt: skip


@pytest.fixture(name="values", scope="module")
def _values() -> Callable[[pd.DataFrame], str]:
    namespace: dict[str, Any] = runpy.run_path(str(SCRIPT), run_name="make_tab")
    return namespace["span_values"]


def _spans(markets: list[str]) -> pd.DataFrame:
    # every horizon of each market spans h hours, a stock's five times as many
    rows = [
        {
            "market": market,
            "horizon": horizon,
            "span_median_hours": float(horizon) * (5 if market == "stock" else 1),
        }
        for market in markets
        for horizon in (1, 4, 24)
    ]
    return pd.DataFrame(rows)


def test_the_longest_horizon_gives_every_market_its_span(values: Callable[[pd.DataFrame], str]):
    text = values(_spans(["crypto", "forex", "commodity", "index", "stock"]))
    assert r"\newcommand{\spanHorizon}{24}" in text
    assert r"\newcommand{\spanCrypto}{24}" in text
    assert r"\newcommand{\spanStock}{120}" in text
    assert r"\newcommand{\spanStockDays}{5.0}" in text


def test_a_missing_market_raises(values: Callable[[pd.DataFrame], str]):
    with pytest.raises(ValueError, match="no span"):
        values(_spans(["crypto", "forex"]))
