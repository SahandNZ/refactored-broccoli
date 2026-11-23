"""Exploratory decomposition of 08-closures (experiments/11-closures-length).

`length_tests` compares, on the closed markets, the benchmark input (the last 512 own bars)
with trading hours only (the own bars of the last 512 calendar hours): the same encoding,
real closes only, at two context lengths; `length_by_market` gives the same skill per market.
Modules import each other relatively, so the package works under any module name.
"""

from ._length import (
    BENCHMARK_INPUT,
    CLOSED,
    TRADING_HOURS,
    LengthSettings,
    closure_day_series,
    length_by_market,
    length_tests,
    read_length_settings,
)

__all__ = [
    "BENCHMARK_INPUT",
    "CLOSED",
    "TRADING_HOURS",
    "LengthSettings",
    "closure_day_series",
    "length_by_market",
    "length_tests",
    "read_length_settings",
]
