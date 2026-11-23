"""Q1: the calendar span of a horizon of h own bars, per market."""

import pandas as pd

from src.benchmark import ForecastCache

NANOSECONDS_PER_HOUR = 3_600_000_000_000
"""Nanoseconds in one hour."""


def horizon_spans(cache: ForecastCache, *, windows: list[str]) -> pd.DataFrame:
    """Return the calendar span of each horizon per market: median and quartiles, in hours.

    The span of a decision is its label time (the close of the label's last own bar) minus the
    decision time; it counts closed hours, so a horizon of h own bars spans h hours on a market
    that never closes and more on one that does.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache (its labels and assets).
    windows : list of str
        The windows whose decisions are measured.

    Returns
    -------
    pd.DataFrame
        ``market``, ``horizon``, ``decisions``, ``span_median_hours``, ``span_q25_hours`` and
        ``span_q75_hours``, one row per market and horizon.

    Raises
    ------
    ValueError
        If no label is in `windows`.
    """
    labels = cache.labels[cache.labels["window"].isin(windows)]
    if labels.empty:
        raise ValueError(f"no label of {cache.folder} in {windows}")
    labels = labels.merge(cache.assets[["asset", "market"]], on="asset", validate="many_to_one")
    nanoseconds = labels["label_time"].to_numpy("datetime64[ns]") - labels["time"].to_numpy(
        "datetime64[ns]"
    )
    hours = nanoseconds.astype("int64") / NANOSECONDS_PER_HOUR
    grouped = labels.assign(hours=hours).groupby(["market", "horizon"])["hours"]
    return pd.DataFrame(
        {
            "decisions": grouped.size(),
            "span_median_hours": grouped.median(),
            "span_q25_hours": grouped.quantile(0.25),
            "span_q75_hours": grouped.quantile(0.75),
        }
    ).reset_index()
