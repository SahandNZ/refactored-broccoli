import numpy as np
import pandas as pd

from src.data._config import Instrument, QualityLimits
from src.enums import Market
from src.utils import active_days, days_after, expected_days, to_utc


def exchange_calendar(instrument: Instrument):
    """Return the exchange calendar an instrument's coverage is measured on, or None.

    The calendar is ``Instrument.calendar`` (``configs/data.toml``: a stock's from its ticker
    suffix, others from their group's ``calendars``); None means coverage over weekdays, or
    every day for crypto.

    Raises
    ------
    ValueError
        If a stock has no calendar (every stock trades on an exchange).
    """
    if instrument.calendar is None and instrument.market is Market.STOCK:
        raise ValueError(f"stock {instrument.name} has no exchange calendar")
    if instrument.calendar is None:
        return None
    import exchange_calendars as xc  # noqa: PLC0415 (heavy optional dependency)

    return xc.get_calendar(instrument.calendar, start="2019-01-01")


def stock_calendar(instrument: Instrument):
    """Return a stock's exchange calendar (`exchange_calendar`), never None.

    Raises
    ------
    ValueError
        If the instrument has no calendar.
    """
    calendar = exchange_calendar(instrument)
    if calendar is None:
        raise ValueError(f"{instrument.name} has no exchange calendar")
    return calendar


def _coverage(times: pd.DatetimeIndex, instrument: Instrument, *, end: str) -> float:
    """Share of the expected days from the first bar to `end` (exclusive) that have a bar.

    Expected days are the sessions of the instrument's exchange calendar, else every weekday
    (every day for crypto).
    """
    first, last_day = times[0], days_after(end, -1)
    calendar = exchange_calendar(instrument)
    if calendar is not None:
        sessions = calendar.sessions_in_range(str(first.date()), str(last_day.date()))
        local = times.tz_convert(calendar.tz).normalize().tz_localize(None).unique()
        return len(sessions.intersection(local)) / max(len(sessions), 1)
    weekdays_only = instrument.market is not Market.CRYPTO
    expected = expected_days(first, last_day, weekdays_only=weekdays_only)
    return len(active_days(times).intersection(expected)) / max(len(expected), 1)


def adjust_for_splits(bars: pd.DataFrame, splits: pd.Series, *, timezone: str) -> pd.DataFrame:
    """Return `bars` adjusted for stock splits, so the series has no split jump.

    Prices before each split are divided by its ratio and volumes multiplied by it.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars with a UTC index; price columns end in open, high, low or close (and ``spread``),
        volume columns contain ``volume``.
    splits : pd.Series
        Split ratio (e.g. 10.0 for 10-for-1) indexed by the local ex-date.
    timezone : str
        The exchange's time zone; a split takes effect at local midnight of its ex-date.

    Returns
    -------
    pd.DataFrame
        The adjusted copy.
    """
    result = bars.copy()
    prices = [
        c for c in bars.columns if c.endswith(("open", "high", "low", "close")) or c == "spread"
    ]
    volumes = [c for c in bars.columns if "volume" in c]
    # a fractional ratio makes volumes fractional, so they cannot stay integer columns
    result[volumes] = result[volumes].astype(float)
    for date, ratio in splits.items():
        effective = pd.Timestamp(date).tz_localize(timezone).tz_convert("UTC")
        before = result.index < effective
        result.loc[before, prices] = result.loc[before, prices] / ratio
        result.loc[before, volumes] = result.loc[before, volumes] * ratio
    return result


def session_closes(bars: pd.DataFrame, calendar) -> pd.Series:
    """Return the last close of each exchange session, from bars inside the session.

    Parameters
    ----------
    bars : pd.DataFrame
        Bars with a UTC index and a ``mid_close`` or ``close`` column.
    calendar : exchange_calendars.ExchangeCalendar
        The exchange.

    Returns
    -------
    pd.Series
        Close per session date (``datetime.date``), sessions without a bar left out.
    """
    price = bars["mid_close"] if "mid_close" in bars.columns else bars["close"]
    schedule = calendar.schedule.loc[str(bars.index[0].date()) : str(bars.index[-1].date())]
    opens = schedule["open"].dt.tz_convert(None).to_numpy()
    closes = schedule["close"].dt.tz_convert(None).to_numpy()
    times = bars.index.tz_convert(None).to_numpy()
    position = np.searchsorted(opens, times, side="right") - 1
    valid = position >= 0
    inside = np.zeros(len(bars), dtype=bool)
    inside[valid] = times[valid] < closes[position[valid]]
    sessions = pd.Index(schedule.index.date)[position[inside]]
    return price[inside].groupby(sessions).last()


def yahoo_return_difference(ours: pd.Series, reference: pd.Series) -> pd.Series:
    """Absolute difference, in bps, between our and Yahoo's daily close-to-close log returns.

    Both series are cut to their common days, and each return runs from one common day to the
    next. Returns, not price levels, are compared, so an offset of the whole level (a different
    adjustment of past prices) does not count; a day where the two disagree (a corporate
    action, a bad close) moves only the one or two returns around it, which the 99th
    percentile of a few hundred days leaves out.

    Parameters
    ----------
    ours : pd.Series
        Our session closes by date (`session_closes`).
    reference : pd.Series
        Yahoo's daily closes by date.

    Returns
    -------
    pd.Series
        Absolute return difference in bps, one value per common day after the first.
    """
    both = pd.concat([ours.rename("ours"), reference.rename("yahoo")], axis=1).dropna()
    returns = pd.DataFrame(np.log(both.to_numpy()), index=both.index, columns=both.columns).diff()
    return ((returns["ours"] - returns["yahoo"]).abs() * 1e4).dropna()


def check_quality(
    bars: pd.DataFrame,
    instrument: Instrument,
    *,
    limits: QualityLimits,
    reference: pd.Series | None,
    end: str,
) -> dict[str, object]:
    """Measure the quality of one instrument's base bars (15m, or 1h) and decide pass or fail.

    Coverage and gaps are measured from the first bar to `end`, so bars that stop early are
    seen: the days after the last bar count as missing and the time from the last bar to `end`
    counts as a gap.

    Parameters
    ----------
    bars : pd.DataFrame
        15m bars (stocks already split-adjusted).
    instrument : Instrument
        The instrument.
    limits : QualityLimits
        The pass thresholds (from ``configs/data.toml``).
    reference : pd.Series or None
        Yahoo daily closes by date for stocks (split-adjusted, not dividend-adjusted); None for
        other markets.
    end : str
        End of the period the bars should cover (exclusive), e.g. ``"2024-01-01"``.

    Returns
    -------
    dict
        coverage, max_gap_days, flat_share, negative_spread_share, outliers (15m log returns
        above ``outlier_sigma`` robust sigmas), max_abs_return, the Yahoo cross-check
        (``yahoo_return_median_bps`` and ``yahoo_return_p99_bps``, `yahoo_return_difference`),
        ``passed`` and ``reason``.
    """
    coverage = _coverage(bars.index, instrument, end=end)
    times = bars.index.append(pd.DatetimeIndex([to_utc(end)]))
    gaps = times.to_series().diff().dt.total_seconds() / 86400
    prefix = "mid_" if "mid_close" in bars.columns else ""
    flat = float((bars[f"{prefix}high"] == bars[f"{prefix}low"]).mean())
    negative = float((bars["spread"] < 0).mean()) if "spread" in bars.columns else 0.0
    returns = np.log(bars[f"{prefix}close"]).diff().dropna()
    sigma = 1.4826 * float((returns - returns.median()).abs().median()) or float(returns.std())
    row: dict[str, object] = {
        "coverage": round(coverage, 4),
        "max_gap_days": round(float(gaps.max()), 2),
        "flat_share": round(flat, 4),
        "negative_spread_share": round(negative, 6),
        "outliers": int((returns.abs() > limits.outlier_sigma * sigma).sum()),
        "max_abs_return": round(float(returns.abs().max()), 4),
        "yahoo_return_median_bps": None,
        "yahoo_return_p99_bps": None,
    }
    reasons: list[str] = []
    if coverage < limits.min_coverage:
        reasons.append(f"coverage {coverage:.3f} < {limits.min_coverage}")
    if row["max_gap_days"] > limits.max_gap_days:  # type: ignore[operator]
        reasons.append(f"gap {row['max_gap_days']} days > {limits.max_gap_days}")
    if flat > limits.max_flat_share:
        reasons.append(f"high equals low in {flat:.1%} of bars")
    if negative > limits.max_negative_spread_share:
        reasons.append(f"ask < bid in {negative:.4%} of bars")
    if instrument.market is not Market.CRYPTO and row["max_abs_return"] > limits.max_abs_return:  # type: ignore[operator]
        reasons.append(f"one-bar jump of {row['max_abs_return']} log return")
    if reference is not None:
        ours = session_closes(bars, stock_calendar(instrument))
        difference = yahoo_return_difference(ours, reference)
        median = round(float(difference.median()), 1)
        p99 = round(float(difference.quantile(0.99)), 1)
        row["yahoo_return_median_bps"], row["yahoo_return_p99_bps"] = median, p99
        if median > limits.yahoo_return_median_bps:
            reasons.append(f"Yahoo median return difference {median} bps")
        if p99 > limits.yahoo_return_p99_bps:
            reasons.append(f"Yahoo 99th percentile return difference {p99} bps")
    row["passed"] = not reasons
    row["reason"] = "; ".join(reasons)
    return row
