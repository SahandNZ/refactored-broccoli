import numpy as np
import pandas as pd

from src.utils import days_after, expected_days, to_utc, week_starts


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Return row-to-row log returns of each column, NaN where a close or the one before is missing.

    Parameters
    ----------
    closes : pd.DataFrame
        Close prices on one shared clock, shape ``(n_times, n_assets)``; NaN where an asset has
        no close (a closed market).

    Returns
    -------
    pd.DataFrame
        Log returns, same shape. A return across a closed period is NaN, never a jump.
    """
    logs = pd.DataFrame(np.log(closes.to_numpy()), index=closes.index, columns=closes.columns)
    return logs.diff()


def weekly_returns(closes: dict[str, pd.Series], *, start: str, end: str) -> pd.DataFrame:
    """Weekly log returns of every asset on every ISO week (UTC) of ``[start, end)``.

    The weekly close is the asset's last daily close of the week (Monday 00:00 to Sunday 24:00
    UTC), within the window; a week without a close has no return and neither has the week
    after it. Weekly closes remove the time-zone asynchrony of daily closes (Tokyo closes before
    New York opens) and keep the Monday moves of markets closed at weekends.

    Parameters
    ----------
    closes : dict of str to pd.Series
        Daily closes of each asset, indexed by UTC day (midnight).
    start, end : str
        The window ``[start, end)``.

    Returns
    -------
    pd.DataFrame
        Log returns, shape ``(n_weeks, n_assets)``, indexed by the week's start, NaN where an
        asset has no return.
    """
    days = expected_days(to_utc(start), days_after(end, -1), weekdays_only=False)
    daily = pd.DataFrame({symbol: series.reindex(days) for symbol, series in closes.items()})
    # groupby's last skips NaN: the last close of each week, NaN for a week without one
    weekly = daily.groupby(week_starts(days)).last()
    return log_returns(weekly)


def correlation_matrix(returns: pd.DataFrame, *, min_overlap: int) -> pd.DataFrame:
    """Pearson correlation of every pair of columns, each pair on the rows both have a return.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns, shape ``(n_times, n_assets)``, NaN where an asset has none.
    min_overlap : int
        Least number of common returns of a pair.

    Returns
    -------
    pd.DataFrame
        Correlation matrix, shape ``(n_assets, n_assets)``.

    Raises
    ------
    ValueError
        If an asset's returns have zero variance (a frozen price has no correlation), or a
        pair has fewer than `min_overlap` common returns (no pair gets a made-up value).
    """
    frozen = returns.columns[returns.std() == 0].tolist()
    if frozen:
        raise ValueError(f"zero variance: the returns of {frozen} never change")
    correlation = returns.corr(method="pearson", min_periods=min_overlap)
    rows, columns = np.nonzero(np.triu(correlation.isna().to_numpy(), k=1))
    if len(rows):
        names = correlation.columns.to_numpy()
        examples = list(zip(names[rows[:5]].tolist(), names[columns[:5]].tolist(), strict=True))
        raise ValueError(
            f"{len(rows)} pairs have fewer than {min_overlap} common returns, e.g. {examples}"
        )
    return correlation
