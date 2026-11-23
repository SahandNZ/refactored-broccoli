import numpy as np
import pandas as pd

from src.enums import Timeframe

# Every bar opens on a fixed UTC grid: intraday, 1d and 3d bars count from 2020-01-01 00:00 UTC
# (a midnight, so 4h bars open at 00, 04, ... and 12h bars at 00, 12); weekly bars count from
# Monday 2019-12-30 00:00 UTC, so every week opens on a Monday.
_GRID_ORIGIN = "2020-01-01"
_WEEK_ORIGIN = "2019-12-30"
_NANOSECONDS_PER_HOUR = 3_600_000_000_000


def to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return `value` as a timezone-aware UTC timestamp.

    Parameters
    ----------
    value : str or pd.Timestamp
        A date or time; a naive value is read as UTC.

    Returns
    -------
    pd.Timestamp
        The same instant in UTC.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def days_after(value: str | pd.Timestamp, days: int) -> pd.Timestamp:
    """Return the UTC time `days` days after `value` (before it when `days` is negative).

    Parameters
    ----------
    value : str or pd.Timestamp
        A date or time; a naive value is read as UTC.
    days : int
        Whole days to add.

    Returns
    -------
    pd.Timestamp
        The shifted UTC time, e.g. ``days_after("2024-01-01", -1)`` is 2023-12-31 00:00 UTC.
    """
    return to_utc(value) + pd.Timedelta(days=days)


def in_window(index: pd.DatetimeIndex, *, start: str, end: str) -> np.ndarray:
    """Return which times of `index` fall in the UTC window ``[start, end)``.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Times, shape ``(n,)``; a naive index (for example dates) is read as UTC.
    start, end : str
        The window's UTC dates or times.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(n,)``.
    """
    times = index.tz_localize("UTC") if index.tz is None else index
    return np.asarray((times >= to_utc(start)) & (times < to_utc(end)))


def ms_to_utc(milliseconds: pd.Series) -> pd.DatetimeIndex:
    """Convert epoch milliseconds to a UTC ``DatetimeIndex`` named ``open_time``.

    Parameters
    ----------
    milliseconds : pd.Series
        Integer epoch milliseconds, shape ``(n,)``.

    Returns
    -------
    pd.DatetimeIndex
        UTC times, shape ``(n,)``.
    """
    index = pd.DatetimeIndex(pd.to_datetime(milliseconds.astype("int64"), unit="ms", utc=True))
    index.name = "open_time"
    return index


def iso_to_utc(values: pd.Series) -> pd.DatetimeIndex:
    """Convert ISO-8601 time strings to a UTC ``DatetimeIndex`` named ``open_time``.

    For example ``"2020-01-02T09:00:00Z"``.

    Parameters
    ----------
    values : pd.Series
        ISO-8601 strings with a zone, shape ``(n,)``.

    Returns
    -------
    pd.DatetimeIndex
        UTC times, shape ``(n,)``.
    """
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True), name="open_time")


def is_utc_index(index: pd.Index) -> bool:
    """Return whether `index` is a timezone-aware UTC ``DatetimeIndex``."""
    return isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"


def bar_grid(timeframe: Timeframe) -> tuple[str, pd.Timestamp]:
    """Return the pandas frequency and the UTC origin of the bar grid of `timeframe`.

    Parameters
    ----------
    timeframe : Timeframe
        The bar length.

    Returns
    -------
    tuple of (str, pd.Timestamp)
        For example ``("4h", Timestamp("2020-01-01 00:00", tz="UTC"))``.

    Raises
    ------
    TypeError
        If `timeframe` is not a `Timeframe`.
    """
    if not isinstance(timeframe, Timeframe):
        raise TypeError(f"timeframe must be a Timeframe, got {timeframe!r}")
    origin = _WEEK_ORIGIN if timeframe is Timeframe.W1 else _GRID_ORIGIN
    return timeframe.pandas_freq, to_utc(origin)


def years_between(start: str, end: str) -> list[tuple[int, str, str]]:
    """Split ``[start, end)`` into calendar years.

    Parameters
    ----------
    start, end : str
        UTC dates.

    Returns
    -------
    list of (int, str, str)
        ``(year, first day, day after the last)`` for each year, cut to ``[start, end)``.

    Examples
    --------
    >>> years_between("2025-06-01", "2026-02-01")
    [(2025, '2025-06-01', '2026-01-01'), (2026, '2026-01-01', '2026-02-01')]
    """
    first, last = to_utc(start), to_utc(end)
    chunks: list[tuple[int, str, str]] = []
    for year in range(first.year, last.year + 1):
        chunk_start = max(first, to_utc(f"{year}-01-01"))
        chunk_end = min(last, to_utc(f"{year + 1}-01-01"))
        if chunk_start < chunk_end:
            chunks.append((year, f"{chunk_start:%Y-%m-%d}", f"{chunk_end:%Y-%m-%d}"))
    return chunks


def months_between(start: str, end: str) -> list[str]:
    """Return the months ``"YYYY-MM"`` that overlap ``[start, end)``.

    Examples
    --------
    >>> months_between("2025-11-15", "2026-02-01")
    ['2025-11', '2025-12', '2026-01']
    """
    first, last = to_utc(start), to_utc(end) - pd.Timedelta(1)
    periods = pd.period_range(first.tz_convert(None), last.tz_convert(None), freq="M")
    return [str(period) for period in periods]


def days_of_month(month: str) -> list[pd.Timestamp]:
    """Return every UTC day (midnight) of the month ``"YYYY-MM"``."""
    days = pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D", tz="UTC")
    return list(days)


def active_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the distinct UTC days (midnights) on which `index` has at least one time."""
    return pd.DatetimeIndex(index.normalize().unique())


def expected_days(
    start: pd.Timestamp, end: pd.Timestamp, *, weekdays_only: bool
) -> pd.DatetimeIndex:
    """Return the UTC days from `start` to `end` (both included) a market should trade.

    Parameters
    ----------
    start, end : pd.Timestamp
        UTC times; only their days count.
    weekdays_only : bool
        True for markets closed at weekends (Monday to Friday kept), False for 24/7 markets.

    Returns
    -------
    pd.DatetimeIndex
        The expected trading days. Exchange holidays are not removed.
    """
    days = pd.date_range(to_utc(start).normalize(), to_utc(end).normalize(), freq="D")
    return days[days.day_of_week < 5] if weekdays_only else days


def in_session(index: pd.DatetimeIndex, *, timezone: str, opens: str, closes: str) -> np.ndarray:
    """Return which bars open inside the daily session ``[opens, closes)`` of local time.

    Parameters
    ----------
    index : pd.DatetimeIndex
        UTC open times of bars, shape ``(n,)``.
    timezone : str
        The exchange's time zone, e.g. ``"America/New_York"``.
    opens, closes : str
        Local session times, e.g. ``"09:30"`` and ``"16:00"``.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(n,)``. Weekends are not removed (bars do not exist then).

    Examples
    --------
    US regular session: ``in_session(index, timezone="America/New_York", opens="09:30",
    closes="16:00")``.
    """
    local = index.tz_convert(timezone)
    minutes = local.hour * 60 + local.minute
    start = pd.Timedelta(f"{opens}:00").total_seconds() // 60
    end = pd.Timedelta(f"{closes}:00").total_seconds() // 60
    return np.asarray((minutes >= start) & (minutes < end))


def hour_grid(start: pd.Timestamp, end: pd.Timestamp, *, step_hours: int) -> pd.DatetimeIndex:
    """Return the UTC times from `start` (included) to `end` (excluded) every `step_hours` hours.

    Parameters
    ----------
    start, end : pd.Timestamp
        UTC times.
    step_hours : int
        Hours between two times (>= 1).

    Returns
    -------
    pd.DatetimeIndex
        The times, shape ``(n,)``.

    Raises
    ------
    ValueError
        If `step_hours` is below 1.
    """
    if step_hours < 1:
        raise ValueError(f"step_hours must be >= 1, got {step_hours}")
    return pd.date_range(to_utc(start), to_utc(end), freq=f"{step_hours}h", inclusive="left")


def to_nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return epoch nanoseconds (int64) of a UTC index, whatever its stored resolution.

    pandas may store times in seconds, milliseconds, microseconds or nanoseconds; ``asi8``
    returns the stored unit, so two indexes are only comparable after this conversion.
    """
    return index.as_unit("ns").asi8


def last_at_or_before(index: pd.DatetimeIndex, times: pd.DatetimeIndex) -> np.ndarray:
    """Return the position in `index` of its last time at or before each of `times`.

    Used to find the last bar that had closed by a decision time (no look-ahead).

    Parameters
    ----------
    index : pd.DatetimeIndex
        Sorted UTC times, e.g. the close times of an asset's bars, shape ``(n,)``.
    times : pd.DatetimeIndex
        UTC times, shape ``(m,)``.

    Returns
    -------
    np.ndarray
        Positions into `index`, shape ``(m,)``; -1 where `index` has no time at or before.
    """
    return np.searchsorted(to_nanoseconds(index), to_nanoseconds(times), side="right") - 1


def _gap_hours(times: pd.DatetimeIndex) -> np.ndarray:
    # the gap in hours from each time to the next, shape (n - 1,)
    return np.diff(to_nanoseconds(times)) / _NANOSECONDS_PER_HOUR


def spacing_hours(times: pd.DatetimeIndex) -> float:
    """Return the median gap in hours between consecutive times of a sorted series.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Sorted UTC times, at least two.

    Returns
    -------
    float
        The median gap, e.g. 1.0 for hourly decisions.

    Raises
    ------
    ValueError
        If there are fewer than two times.
    """
    if len(times) < 2:
        raise ValueError(f"a spacing needs at least 2 times, got {len(times)}")
    return float(np.median(_gap_hours(times)))


def is_weekend(index: pd.DatetimeIndex) -> np.ndarray:
    """Return which times fall on a Saturday or a Sunday (UTC).

    Parameters
    ----------
    index : pd.DatetimeIndex
        UTC times, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Booleans, shape ``(n,)``.
    """
    return np.asarray(index.tz_convert("UTC").day_of_week >= 5)


def is_week_start(index: pd.DatetimeIndex) -> np.ndarray:
    """Return which times fall on a Sunday or a Monday (UTC): when a market reopens after a weekend.

    Parameters
    ----------
    index : pd.DatetimeIndex
        UTC times, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Booleans, shape ``(n,)``.
    """
    return np.isin(index.tz_convert("UTC").day_of_week, (6, 0))


def week_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the start of each time's ISO week: the Monday 00:00 UTC on or before it.

    Parameters
    ----------
    index : pd.DatetimeIndex
        UTC times, shape ``(n,)``.

    Returns
    -------
    pd.DatetimeIndex
        Week starts, shape ``(n,)``, e.g. 2024-01-01 00:00 UTC for Sunday 2024-01-07 23:00.
    """
    utc = index.tz_convert("UTC")
    return pd.DatetimeIndex(utc.normalize() - pd.to_timedelta(utc.day_of_week, unit="D"))


def closures(close_times: pd.DatetimeIndex, *, bar: pd.Timedelta, min_hours: float) -> pd.DataFrame:
    """Find every closure of a market: at least `min_hours` hours without a bar.

    A closure runs from the close of a bar to the open of the next one (its close time minus
    `bar`); the bar after it reopens the market.

    Parameters
    ----------
    close_times : pd.DatetimeIndex
        Sorted UTC close times of the market's bars, shape ``(n,)``.
    bar : pd.Timedelta
        The length of one bar.
    min_hours : float
        The shortest closure that counts, in hours.

    Returns
    -------
    pd.DataFrame
        One row per closure: ``position`` (of the first bar after it in `close_times`, at
        least 1), ``close_time`` (the close of the last bar before it), ``reopen_time`` (the
        open of the first bar after it) and ``closed_hours``.

    Examples
    --------
    A market that closes on Friday 21:00 and reopens on Sunday 22:00 is closed 49 hours.
    """
    closed_hours = _gap_hours(close_times) - bar.value / _NANOSECONDS_PER_HOUR
    positions = np.flatnonzero(closed_hours >= min_hours) + 1
    return pd.DataFrame(
        {
            "position": positions,
            "close_time": close_times[positions - 1],
            "reopen_time": close_times[positions] - bar,
            "closed_hours": closed_hours[positions - 1],
        }
    )


def half_years(times: pd.DatetimeIndex) -> np.ndarray:
    """Return the calendar half-year of each UTC time.

    Examples
    --------
    >>> half_years(pd.DatetimeIndex(["2024-03-01", "2024-07-01"], tz="UTC")).tolist()
    ['2024H1', '2024H2']
    """
    halves = np.where(times.month <= 6, "H1", "H2")
    return np.char.add(times.year.to_numpy().astype(str), halves)
