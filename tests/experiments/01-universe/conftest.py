"""Fixtures of the 01-universe tests: its ``lib`` package and fake bars."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from src import LAB_DIR
from src.data import Instrument, add_mid_and_spread, bars_file
from src.enums import Market

STUDY_DIR = LAB_DIR / "experiments" / "01-universe"

type WriteBars = Callable[[Path, Instrument, pd.Series, float], None]


def _write_bars(
    data_dir: Path, instrument: Instrument, closes: pd.Series, liquidity: float
) -> None:
    """Write `closes` as the base bars of `instrument`, with the columns of its source.

    Crypto has a quote volume (Binance), a US stock a vwap and volume (Alpaca), the rest bid
    and ask (Dukascopy); a higher `liquidity` means more traded value or a tighter spread.
    """
    frame = pd.DataFrame(index=pd.DatetimeIndex(closes.index, name="open_time"))
    if instrument.market is Market.CRYPTO:
        frame["open"], frame["high"], frame["low"], frame["close"] = (
            closes, closes * 1.001, closes * 0.999, closes,
        )  # fmt: skip
        frame["amount"] = 1e6 * liquidity / 24
    elif instrument.market is Market.STOCK and "." not in instrument.name:
        frame["open"], frame["high"], frame["low"], frame["close"] = (
            closes, closes * 1.001, closes * 0.999, closes,
        )  # fmt: skip
        frame["vwap"], frame["volume"] = closes, 1e4 * liquidity
    else:
        half = closes * 0.5e-4 / liquidity
        for side, sign in (("bid", -1), ("ask", 1)):
            frame[f"{side}_open"] = frame[f"{side}_close"] = closes + sign * half
            frame[f"{side}_high"] = closes * 1.001 + sign * half
            frame[f"{side}_low"] = closes * 0.999 + sign * half
        frame = add_mid_and_spread(frame)
    path = bars_file(data_dir, instrument.market, instrument.name, instrument.timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


@pytest.fixture(name="universe", scope="session")
def _universe(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)


@pytest.fixture(name="write_bars", scope="session")
def _write_bars_fixture() -> WriteBars:
    return _write_bars


@pytest.fixture(name="options")
def _options() -> dict[str, Any]:
    """Study options for two months of selection and one held-out month (a fresh copy)."""
    return {
        "start": "2024-01-01", "end": "2024-03-01", "holdout_end": "2024-04-01",
        "start_tolerance_days": 7, "listed_by": {"crypto": "2024-01-10"},
        "pegged_currencies": ["HKD", "DKK"], "min_daily_amount": 1.0, "max_spread_ratio": 3.0,
        "min_overlap": 10, "holdout_min_overlap": 5, "bootstrap_min_overlap": 5,
        "duplicate_threshold": 0.85,
        "max_per_sector": 3, "random_universes": 5, "q1_percentile": 90,
        "bootstrap_block_weeks": 2, "bootstrap_resamples": 20, "interval_level": 0.9,
        "anchors": {
            "crypto": ["AAA"], "forex": ["EURUSD"], "commodity": [], "stock_us": [],
            "stock_non_us": [],
        },
        "quotas": {"crypto": 2, "forex": 2, "commodity": 8, "stock_us": 1, "stock_non_us": 1},
        "liquid_pool_factor": {
            "crypto": 2, "forex": 2, "commodity": 2, "stock_us": 2, "stock_non_us": 2,
        },
    }  # fmt: skip
