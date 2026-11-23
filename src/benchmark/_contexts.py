from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.enums import Timeframe
from src.utils import hour_grid, last_at_or_before, to_nanoseconds

HOUR = Timeframe.H1.timedelta
"""The length of one bar of the benchmark."""


def context_ends(closes: pd.Series, times: pd.DatetimeIndex, *, lookback: int) -> np.ndarray:
    """Find the position of the last close of each decision's context (its own bars only).

    The context of a decision at time ``t`` is the asset's last `lookback` closes that had
    closed by ``t`` (no look-ahead). A closed market has no bars, so nothing is filled. A context
    is invalid when the asset has fewer than `lookback` closes by ``t``. Its freshness needs no
    check: the benchmark keeps one decision per new own close (`decision_labels`), and
    08-closures decides one hour before each own close.

    Parameters
    ----------
    closes : pd.Series
        1h closes of one asset, indexed by the bars' close times (UTC), oldest first.
    times : pd.DatetimeIndex
        Decision times, shape ``(n,)``.
    lookback : int
        Closes per context.

    Returns
    -------
    np.ndarray
        Position in `closes` of each context's last close, shape ``(n,)``; -1 where the context
        is invalid. The context is ``closes.iloc[end - lookback + 1 : end + 1]``.
    """
    ends = last_at_or_before(pd.DatetimeIndex(closes.index), times)
    return np.where(ends >= lookback - 1, ends, -1)


@dataclass(frozen=True, slots=True)
class CalendarWindows:
    """The last calendar hours up to each decision of one asset (`calendar_windows`).

    Attributes
    ----------
    values : np.ndarray
        Each hour's last close at or before it (forward-filled), shape ``(n, lookback)``,
        oldest first.
    is_closed : np.ndarray
        Whether no own bar closed at the hour, shape ``(n, lookback)``.
    hours : pd.DatetimeIndex
        The hours themselves, row after row, shape ``(n * lookback,)``.
    is_valid : np.ndarray
        Whether the asset has `lookback` hours of history at the decision, shape ``(n,)``.
    """

    values: np.ndarray
    is_closed: np.ndarray
    hours: pd.DatetimeIndex
    is_valid: np.ndarray


def calendar_windows(closes: pd.Series, times: pd.DatetimeIndex, lookback: int) -> CalendarWindows:
    """Cut the `lookback` calendar hours up to each decision time from one asset's closes.

    Every calendar hour from the asset's first close holds the last close at or before it and
    whether no bar closed at it; nothing after a decision time enters its window.

    Parameters
    ----------
    closes : pd.Series
        1h closes of one asset, indexed by the bars' close times (UTC, on the hour).
    times : pd.DatetimeIndex
        Decision times, shape ``(n,)``.
    lookback : int
        Hours per window.

    Returns
    -------
    CalendarWindows
        The windows; a row without `lookback` hours of history is filled with the first
        hours and marked invalid.
    """
    close_times = pd.DatetimeIndex(closes.index)
    grid = hour_grid(close_times[0], close_times[-1] + HOUR, step_hours=1)
    last = last_at_or_before(close_times, grid)
    is_closed = to_nanoseconds(close_times[last]) != to_nanoseconds(grid)
    filled = closes.to_numpy(dtype=float)[last]
    ends = last_at_or_before(grid, times)
    is_valid = ends >= lookback - 1
    rows = np.where(is_valid, ends, lookback - 1)[:, None] + np.arange(1 - lookback, 1)
    return CalendarWindows(filled[rows], is_closed[rows], grid[rows.ravel()], is_valid)


def own_bars_only(values: np.ndarray, is_closed: np.ndarray) -> np.ndarray:
    """Keep the own bars of each window, in order, right-aligned and NaN-padded on the left.

    The "trading hours only" representation of a calendar window (08-closures, 07-rq5-off-
    hours): a stable sort puts the closed hours first and keeps the order of the rest, then
    the closed hours become NaN.

    Parameters
    ----------
    values : np.ndarray
        Hourly values, shape ``(n, lookback)``.
    is_closed : np.ndarray
        Which hours had no own bar, shape ``(n, lookback)``.

    Returns
    -------
    np.ndarray
        Shape ``(n, lookback)``; the first ``is_closed.sum(axis=1)`` values of each row NaN.
    """
    order = np.argsort(~is_closed, axis=1, kind="stable")
    dropped = np.take_along_axis(values, order, axis=1)
    n_closed: np.ndarray = is_closed.sum(axis=1, keepdims=True)
    width: int = values.shape[1]
    is_padding = np.arange(width) < n_closed
    return np.where(is_padding, np.nan, dropped)
