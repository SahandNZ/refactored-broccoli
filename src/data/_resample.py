import pandas as pd

from src.enums import Timeframe
from src.utils import OHLC_FIELDS, bar_grid, is_utc_index


def _aggregation(column: str) -> str:
    if column.endswith("open"):
        return "first"
    if column.endswith("high"):
        return "max"
    if column.endswith("low"):
        return "min"
    if column.endswith("close"):
        return "last"
    return "sum"


def add_mid_and_spread(bars: pd.DataFrame) -> pd.DataFrame:
    """Return `bars` with mid prices and the closing spread computed from bid and ask.

    Parameters
    ----------
    bars : pd.DataFrame
        Columns ``bid_open, bid_high, bid_low, bid_close`` and the same for ``ask``.

    Returns
    -------
    pd.DataFrame
        A copy with ``mid_open, mid_high, mid_low, mid_close`` (the mean of bid and ask) and
        ``spread`` (``ask_close - bid_close``) added.
    """
    result = bars.copy()
    for field in OHLC_FIELDS:
        result[f"mid_{field}"] = (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2
    result["spread"] = bars["ask_close"] - bars["bid_close"]
    return result


def resample_bars(bars: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate bars into coarser bars on the fixed UTC grid of `timeframe`.

    A bar is labelled by its open time and covers ``[open, open + length)``. Periods without
    any input bar (a closed market) produce no bar: they are absent, never filled.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars with a UTC ``DatetimeIndex`` of open times, finer than or equal to `timeframe`.
        Columns ending in ``open``, ``high``, ``low`` and ``close`` aggregate as first, max,
        min and last; ``vwap`` is the volume-weighted mean of the finer vwaps; every other
        column (volumes, ``n_bars``) is summed. Mid prices and ``spread`` are recomputed from
        bid and ask when both are present.
    timeframe : Timeframe
        The bar length of the result.

    Returns
    -------
    pd.DataFrame
        One row per non-empty bar, indexed by its UTC open time.

    Raises
    ------
    TypeError
        If `timeframe` is not a `Timeframe`.
    ValueError
        If the index is not timezone-aware UTC.
    """
    frequency, origin = bar_grid(timeframe)
    if not is_utc_index(bars.index):
        raise ValueError("bars must have a timezone-aware UTC DatetimeIndex")
    derived = [c for c in bars.columns if c.startswith("mid_") or c == "spread"]
    source = bars.drop(columns=derived)
    # pandas ignores the origin of day-based frequencies on a timezone-aware index, so the
    # grid is applied on the same instants written as naive UTC and the zone is put back after
    naive = source.set_axis(source.index.tz_convert(None))
    has_vwap = "vwap" in naive.columns
    if has_vwap:
        naive = naive.assign(vwap=naive["vwap"] * naive["volume"])
    grouped = naive.resample(frequency, origin=origin.tz_convert(None), label="left", closed="left")
    result = grouped.agg({column: _aggregation(column) for column in source.columns})
    result = result[grouped.size() > 0]
    if has_vwap:
        result["vwap"] = result["vwap"] / result["volume"]
    result.index = result.index.tz_localize("UTC")
    if "bid_close" in result.columns and "ask_close" in result.columns:
        result = add_mid_and_spread(result)
    result.index.name = "open_time"
    return result
