"""The decisions after a market reopens and their contexts in three representations (08-closures).

`reopening_decisions` finds the closures of an asset and the first own bars after each
reopening, `cut_weekends` makes artificial closures in a market that never closes (the crypto
control), and `closure_contexts` builds, from the same calendar hours, the context of every
decision with trading hours only (the closed hours dropped), forward-filled or missing (NaN),
plus the benchmark's own input (the last own bars, reaching further back), with its label.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.benchmark import HOUR, calendar_windows, context_ends, own_bars_only
from src.panel import ewma_volatility
from src.utils import closures, is_weekend, last_at_or_before, to_nanoseconds

TRADING_HOURS = "trading hours only"
FORWARD_FILLED = "forward-filled"
MISSING = "missing"
REPRESENTATIONS = (TRADING_HOURS, FORWARD_FILLED, MISSING)
"""How the closed hours of the same calendar hours are shown: dropped (trading hours only, the
reference of the tests), forward-filled (made-up flat prices, the object of study) or missing
(NaN)."""

BENCHMARK_INPUT = "benchmark input"
"""The benchmark's own input, the last `lookback` own bars (reaching further back than the
calendar hours): a descriptive arm, never tested."""

ARMS = (*REPRESENTATIONS, BENCHMARK_INPUT)
"""Every context a model forecasts from: the three representations and the benchmark input."""


def cut_weekends(closes: pd.Series) -> pd.Series:
    """Remove every bar that opens on a Saturday or a Sunday (UTC): an artificial closure.

    Parameters
    ----------
    closes : pd.Series
        1h closes of one asset, indexed by the bars' close times (UTC).

    Returns
    -------
    pd.Series
        The closes of the bars that open Monday to Friday.
    """
    open_times = pd.DatetimeIndex(closes.index) - HOUR
    return closes[~is_weekend(open_times)]


def reopening_decisions(
    closes: pd.Series, *, min_closure_hours: float, reopen_bars: int
) -> pd.DataFrame:
    """Find the first `reopen_bars` own bars after every closure and the decision before each.

    A closure is a stretch of at least `min_closure_hours` hours without a bar between two own
    bars (`src.utils.closures`); the bar after it reopens the market. Its target bars are the
    reopening bar and the next own bars, up to `reopen_bars` of them or until the next
    reopening. The decision of a target bar is one hour before its close (when it opens on the
    1h grid; the US half bar 09:30-10:00 New York is decided at 09:00), so
    its label is the next own bar in every representation: one step ahead in calendar time and
    in event time, and one decision per own bar. The decision times follow the exchange's
    schedule (when the market opens), known in advance.

    Parameters
    ----------
    closes : pd.Series
        1h closes of one asset, indexed by the bars' close times (UTC), oldest first.
    min_closure_hours : float
        The shortest closure, in hours without a bar.
    reopen_bars : int
        Target bars per reopening.

    Returns
    -------
    pd.DataFrame
        ``time`` (the decision), ``bar`` (0 for the reopening bar, 1 for the next, ...),
        ``closed_hours`` (the length of the closure before it) and ``closure_day`` (the UTC
        day, at midnight, on which that closure ends: the day of its reopening bar's
        decision); one row per target bar.
    """
    close_times = pd.DatetimeIndex(closes.index)
    found = closures(close_times, bar=HOUR, min_hours=min_closure_hours)
    # the closure before each bar's last reopening (NaN at bars that do not reopen), and the
    # position of that reopening, -1 before the first one
    closed_hours = np.full(len(close_times), np.nan)
    closed_hours[found["position"]] = found["closed_hours"]
    positions = np.arange(len(close_times))
    first = np.maximum.accumulate(np.where(np.isfinite(closed_hours), positions, -1))
    bar = positions - first
    is_target = (first >= 0) & (bar < reopen_bars)
    reopenings = close_times[first[is_target]] - HOUR
    return pd.DataFrame(
        {
            "time": close_times[is_target] - HOUR,
            "bar": bar[is_target],
            "closed_hours": closed_hours[first[is_target]],
            "closure_day": reopenings.normalize(),
        }
    )


@dataclass(frozen=True, slots=True)
class ClosureContexts:
    """The labelled decisions of 08-closures and their contexts in every representation.

    Every representation sees the same `lookback` calendar hours up to the decision, so they
    differ only in how the closed hours are shown (not in how much history they hold).

    Attributes
    ----------
    decisions : pd.DataFrame
        One row per decision: the columns of the given decisions (``asset``, ``time``,
        ``bar``, ``closed_hours``, ``closure_day`` and, from `closure_losses`, ``window``),
        then ``label_time``, ``label``, ``last_close`` and ``sigma`` (as in `src.benchmark`
        labels at horizon 1).
    windows : dict of str to np.ndarray
        Per arm of `ARMS`, the contexts, shape ``(n, lookback)``, oldest first.
        ``forward-filled`` and ``missing`` hold the `lookback` calendar hours;
        ``trading hours only`` the own bars that closed in those hours, right-aligned, its
        first ``lookback - own_bars`` values NaN (a shorter context: only the last
        `own_bars` values are given to a model); ``benchmark input`` the last `lookback` own
        closes, as in the benchmark.
    is_closed : np.ndarray
        Which calendar hours had no own bar, shape ``(n, lookback)``: the made-up values of
        ``forward-filled`` and the NaN of ``missing``.
    own_bars : np.ndarray
        The number of own bars in each window, shape ``(n,)``: the length of
        ``trading hours only``.
    short_contexts : int
        Decisions dropped only because their trading-hours context held fewer than
        ``min_own_closes`` own closes (kept by `select`, a count of the whole group).
    """

    decisions: pd.DataFrame
    windows: dict[str, np.ndarray]
    is_closed: np.ndarray
    own_bars: np.ndarray
    short_contexts: int

    def select(self, rows: np.ndarray) -> "ClosureContexts":
        """Return the decisions marked by the booleans `rows`, shape ``(n,)``, with contexts."""
        windows = {name: values[rows] for name, values in self.windows.items()}
        decisions = self.decisions[rows].reset_index(drop=True)
        return ClosureContexts(
            decisions, windows, self.is_closed[rows], self.own_bars[rows], self.short_contexts
        )


def closure_contexts(
    closes: dict[str, pd.Series],
    decisions: pd.DataFrame,
    *,
    lookback: int,
    span_bars: int,
    min_own_closes: int,
) -> ClosureContexts:
    """Build the label and the contexts of every decision: three from the same calendar hours.

    The label is the benchmark's at horizon 1: with ``end`` the last own close at or before the
    decision time, ``label = log(P[end + 1] / P[end]) / sigma``, ``sigma`` the EWMA volatility
    of own-bar log returns at ``end`` (`src.panel.ewma_volatility`). Every representation reads
    the `lookback` calendar hours up to the decision (`src.benchmark.calendar_windows`):
    forward-filled and missing all of them, trading hours only the own bars that closed in
    them. The benchmark input is the last `lookback` own closes (`src.benchmark.context_ends`).
    A decision is kept when the asset has `lookback` calendar hours and `lookback` own closes
    before it, at least `min_own_closes` own closes in those hours (a data gap can empty
    them; the trading-hours context is then too short) and a finite label, so every arm has
    exactly the same decisions and labels.
    Nothing after the decision time enters a context.

    Parameters
    ----------
    closes : dict of str to pd.Series
        1h closes per asset, indexed by the bars' close times (UTC; on the hour).
    decisions : pd.DataFrame
        ``asset``, ``time`` and any other columns (`reopening_decisions`); each time is one
        hour before the close of the asset's next own bar.
    lookback : int
        Calendar hours per context.
    span_bars : int
        EWMA span of the label's volatility, in bars.
    min_own_closes : int
        The fewest own closes in the `lookback` calendar hours (the length of the
        trading-hours context), between 1 and `lookback`.

    Returns
    -------
    ClosureContexts
        The kept decisions with their labels, and their contexts.

    Raises
    ------
    ValueError
        If there is no decision, `min_own_closes` is not between 1 and `lookback`, or a
        decision's next own bar does not close one hour after it.
    """
    if not 1 <= min_own_closes <= lookback:
        raise ValueError(
            f"min_own_closes must be between 1 and lookback {lookback}, got {min_own_closes}"
        )
    if decisions.empty:
        raise ValueError("closure_contexts needs at least one decision, got none")
    parts: list[pd.DataFrame] = []
    filled: list[np.ndarray] = []
    closed: list[np.ndarray] = []
    own: list[np.ndarray] = []
    short_contexts = 0
    for asset, group in decisions.groupby("asset", sort=False):
        series = closes[str(asset)]
        times = pd.DatetimeIndex(group["time"])
        prices = series.to_numpy(dtype=float)
        ends = last_at_or_before(pd.DatetimeIndex(series.index), times)
        has_own_bars = context_ends(series, times, lookback=lookback) >= 0
        hours = calendar_windows(series, times, lookback)
        sigma = ewma_volatility(series, times, span_bars=span_bars)
        later = np.minimum(ends + 1, len(series) - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            label = np.log(prices[later] / prices[ends]) / sigma
        is_long = (~hours.is_closed).sum(axis=1) >= min_own_closes
        keep = has_own_bars & (ends + 1 < len(series)) & hours.is_valid & np.isfinite(label)
        short_contexts += int((keep & ~is_long).sum())
        keep &= is_long
        label_time = pd.DatetimeIndex(series.index[later[keep]])
        if not np.array_equal(to_nanoseconds(label_time - HOUR), to_nanoseconds(times)[keep]):
            raise ValueError(f"{asset}: every decision must be one hour before an own close")
        kept = group.iloc[np.flatnonzero(keep)]
        parts.append(
            kept.assign(
                label_time=label_time,
                label=label[keep],
                last_close=prices[ends[keep]],
                sigma=sigma[keep],
            )
        )
        filled.append(hours.values[keep])
        closed.append(hours.is_closed[keep])
        own.append(prices[ends[keep, None] + np.arange(1 - lookback, 1)])
    window, is_closed = np.concatenate(filled), np.concatenate(closed)
    windows = {
        TRADING_HOURS: own_bars_only(window, is_closed),
        FORWARD_FILLED: window,
        MISSING: np.where(is_closed, np.nan, window),
        BENCHMARK_INPUT: np.concatenate(own),
    }
    own_bars = (~is_closed).sum(axis=1)
    found = pd.concat(parts, ignore_index=True)
    return ClosureContexts(found, windows, is_closed, own_bars, short_contexts)
