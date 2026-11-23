"""Study 07-rq5-off-hours: the weekends of the closed markets and the crypto move over them.

`read_weekends` finds every weekend of every closed-market asset with the crypto basket over
the same closed hours, Friday's information, the reopening gap (Q2) and the first bar's move
and its vol-scaled label (Q1).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import HOUR
from src.enums import Market, Timeframe
from src.panel import ewma_volatility, hourly_closes, load_panel_bars
from src.utils import closures, is_week_start, last_at_or_before, week_starts

from ._settings import OffHoursSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Weekends:
    """Every weekend of the closed-market assets, and the prices their model contexts come from.

    Attributes
    ----------
    table : pd.DataFrame
        One row per weekend and asset (`weekend_rows` with ``asset`` and ``market``),
        reopening from ``start`` to the end of the scored folds.
    closes : dict of str to pd.Series
        Each asset's 1h mid closes, indexed by the bars' close times.
    crypto : pd.DataFrame
        Log 1h closes of the basket, one column per asset, indexed by close time; NaN where an
        asset has no bar.
    """

    table: pd.DataFrame
    closes: dict[str, pd.Series]
    crypto: pd.DataFrame


def basket_return(
    crypto: pd.DataFrame, start: pd.DatetimeIndex, end: pd.DatetimeIndex
) -> np.ndarray:
    """Return the mean log return of the basket from each of `start` to each of `end`.

    Parameters
    ----------
    crypto : pd.DataFrame
        Log closes, one column per asset, indexed by close time.
    start, end : pd.DatetimeIndex
        Close times, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``; NaN where an asset has no close at exactly one of the two times (a
        missing bar is never filled).
    """
    first = crypto.reindex(start).to_numpy(dtype=float)
    last = crypto.reindex(end).to_numpy(dtype=float)
    return (last - first).mean(axis=1)


def weekend_closures(
    close_times: pd.DatetimeIndex, *, min_closed_hours: float, max_closed_hours: float
) -> pd.DataFrame:
    """Return the weekend closures of an asset: those that reopen on a Sunday or a Monday (UTC).

    A closure of at least `min_closed_hours` hours (`src.utils.closures`) that reopens on
    another day (a midweek holiday, Easter Tuesday) is not a weekend. Nor is one longer than
    `max_closed_hours` (a data outage, a holiday that starts midweek: several days of crypto
    moves in one point); those are logged. Each week (of the close,
    `src.utils.week_starts`) has at most one weekend, its first: a second closure that starts
    in the same week, e.g. a gap in the data right after a Sunday reopening, is left out and
    logged.

    Parameters
    ----------
    close_times : pd.DatetimeIndex
        Sorted UTC close times of the asset's 1h bars.
    min_closed_hours, max_closed_hours : float
        The shortest and the longest closure that count, in hours.

    Returns
    -------
    pd.DataFrame
        The rows of `src.utils.closures` that are weekends.
    """
    found = closures(close_times, bar=HOUR, min_hours=min_closed_hours)
    found = found[is_week_start(pd.DatetimeIndex(found["reopen_time"]))]
    is_long = found["closed_hours"] > max_closed_hours
    if is_long.any():
        logger.info(
            "%d closures left out: longer than %s hours", int(is_long.sum()), max_closed_hours
        )
    found = found[~is_long]
    is_repeat = week_starts(pd.DatetimeIndex(found["close_time"])).duplicated()
    if is_repeat.any():
        logger.info("%d closures left out: a second closure in the week", int(is_repeat.sum()))
    return found[~is_repeat].reset_index(drop=True)


def weekend_rows(
    bars: pd.DataFrame, crypto: pd.DataFrame, settings: OffHoursSettings
) -> pd.DataFrame:
    """Return every weekend of one closed-market asset: its inputs, its gap and its move.

    The closure runs from ``close_time`` (the close of the last bar before it) to
    ``reopen_time`` (the open of the first bar after it). The inputs use only what is known at
    the reopening: Friday's information and the label's scale end at ``close_time`` and the
    crypto move covers exactly the closed hours. The targets are the gap to the first bar's
    open (Q2) and the move to its close at ``label_time`` (Q1).

    Parameters
    ----------
    bars : pd.DataFrame
        The asset's 1h bars with ``mid_open`` and ``mid_close``, indexed by UTC open time.
    crypto : pd.DataFrame
        Log 1h closes of the basket, one column per asset, indexed by close time.
    settings : OffHoursSettings
        The study's settings (``min_closed_hours``, ``max_closed_hours``, ``friday_hours``,
        ``span_bars``).

    Returns
    -------
    pd.DataFrame
        One row per weekend: ``week`` (the Monday 00:00 UTC that starts the week of the
        Friday close: one point per weekend for every asset, however its reopening is
        staggered), ``close_time``, ``reopen_time``, ``label_time``, ``friday_close`` (the
        last mid close), ``sigma`` (the benchmark's EWMA volatility of hourly log returns at
        ``close_time``), ``friday_target`` and ``friday_crypto`` (log returns of the asset and
        the basket over the last ``friday_hours`` hours before the close),
        ``weekend_crypto`` (the basket from ``close_time`` to ``reopen_time``), ``gap`` (log of
        the first open over ``friday_close``), ``move`` (log of the first close over
        ``friday_close``) and ``label`` (``move / sigma``, the benchmark's label at horizon
        1). NaN where a price is missing.
    """
    open_times = pd.DatetimeIndex(bars.index)
    close_times = open_times + HOUR
    weekends = weekend_closures(
        close_times,
        min_closed_hours=settings.min_closed_hours,
        max_closed_hours=settings.max_closed_hours,
    )
    first = weekends["position"].to_numpy()
    last = first - 1
    mid_open = bars["mid_open"].to_numpy(dtype=float)
    mid_close = bars["mid_close"].to_numpy(dtype=float)
    close_time = pd.DatetimeIndex(weekends["close_time"])
    reopen_time = pd.DatetimeIndex(weekends["reopen_time"])
    friday_start = close_time - settings.friday_hours * HOUR
    earlier = last_at_or_before(close_times, friday_start)
    friday_price = np.where(earlier >= 0, mid_close[np.maximum(earlier, 0)], np.nan)
    closes = pd.Series(mid_close, index=close_times)
    sigma = ewma_volatility(closes, close_time, span_bars=settings.span_bars)
    move = np.log(mid_close[first] / mid_close[last])
    return pd.DataFrame(
        {
            "week": week_starts(close_time),
            "close_time": close_time,
            "reopen_time": reopen_time,
            "label_time": reopen_time + HOUR,
            "friday_close": mid_close[last],
            "sigma": sigma,
            "friday_target": np.log(mid_close[last] / friday_price),
            "friday_crypto": basket_return(crypto, friday_start, close_time),
            "weekend_crypto": basket_return(crypto, close_time, reopen_time),
            "gap": np.log(mid_open[first] / mid_close[last]),
            "move": move,
            "label": move / sigma,
        }
    )


def read_weekends(
    data_dir: Path, assets: dict[str, Market], settings: OffHoursSettings
) -> Weekends:
    """Read the 1h bars of the closed-market assets and the basket and find every weekend.

    Parameters
    ----------
    data_dir : Path
        The lab's ``data/`` folder (read only).
    assets : dict of str to Market
        The closed-market assets and their markets (`AssetSets`, both sets).
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    Weekends
        The weekends that reopen from ``start`` to the end of the scored folds, sorted by
        asset and time, with the prices of the model contexts.

    Raises
    ------
    ValueError
        If `assets` is empty.
    """
    if not assets:
        raise ValueError("read_weekends needs at least one closed-market asset, got none")
    basket = dict.fromkeys(settings.crypto_basket, Market.CRYPTO)
    bars = load_panel_bars(data_dir, assets | basket, (Timeframe.H1,))
    closes = hourly_closes(bars)
    crypto = pd.DataFrame({asset: np.log(closes[asset]) for asset in settings.crypto_basket})
    parts = [
        weekend_rows(bars[asset][Timeframe.H1], crypto, settings).assign(
            asset=asset, market=market.value
        )
        for asset, market in assets.items()
    ]
    table = pd.concat(parts, ignore_index=True)
    is_inside = (table["reopen_time"] >= settings.start) & (table["reopen_time"] < settings.end)
    table = table[is_inside].sort_values(["asset", "reopen_time"], ignore_index=True)
    return Weekends(table, {asset: closes[asset] for asset in assets}, crypto)
