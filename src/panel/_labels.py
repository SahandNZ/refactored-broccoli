import numpy as np
import pandas as pd

from src.utils import last_at_or_before


def vol_scaled_returns(
    closes: pd.Series, times: pd.DatetimeIndex, *, horizon_hours: int, span_bars: int
) -> np.ndarray:
    """Forward log returns over `horizon_hours` divided by the volatility known at each time.

    ``y = log(P[t + h] / P[t]) / (sigma_t * sqrt(h))`` where ``P[t]`` is the close of the last
    bar that closed by ``t``, ``P[t + h]`` the same at ``t + h``, and ``sigma_t`` the EWMA
    standard deviation of hourly log returns up to ``t``. The label is NaN when no new bar
    closes between ``t`` and ``t + h`` (a closed market carries no information) or when the
    volatility is not yet known.

    Parameters
    ----------
    closes : pd.Series
        1h closes of one asset, indexed by the bars' close times (UTC); a closed market has no
        rows.
    times : pd.DatetimeIndex
        Decision times, shape ``(n,)``.
    horizon_hours : int
        Forecast horizon.
    span_bars : int
        EWMA span of the volatility, in bars.

    Returns
    -------
    np.ndarray
        Labels, shape ``(n,)``.

    References
    ----------
    B. Lim, S. Zohren, S. Roberts, "Enhancing Time Series Momentum Strategies Using Deep
    Neural Networks", Journal of Financial Data Science 1(4), 2019 (volatility scaling).
    """
    returns = np.log(closes).diff()
    sigma = returns.ewm(span=span_bars, min_periods=span_bars).std().to_numpy()
    index = pd.DatetimeIndex(closes.index)
    now = last_at_or_before(index, times)
    later = last_at_or_before(index, times + pd.Timedelta(hours=horizon_hours))
    values = closes.to_numpy(dtype=float)
    valid = (now >= 0) & (later > now)
    labels = np.full(len(times), np.nan)
    scale = sigma[now[valid]] * np.sqrt(horizon_hours)
    labels[valid] = np.log(values[later[valid]] / values[now[valid]]) / scale
    labels[~np.isfinite(labels)] = np.nan
    return labels


def ewma_volatility(closes: pd.Series, times: pd.DatetimeIndex, *, span_bars: int) -> np.ndarray:
    """EWMA standard deviation of hourly log returns known at each time (the label's scale).

    Parameters
    ----------
    closes : pd.Series
        1h closes indexed by close times (UTC).
    times : pd.DatetimeIndex
        Decision times, shape ``(n,)``.
    span_bars : int
        EWMA span in bars.

    Returns
    -------
    np.ndarray
        Volatility per time, shape ``(n,)``; NaN before enough bars.
    """
    sigma = np.log(closes).diff().ewm(span=span_bars, min_periods=span_bars).std().to_numpy()
    now = last_at_or_before(pd.DatetimeIndex(closes.index), times)
    result = np.full(len(times), np.nan)
    result[now >= 0] = sigma[now[now >= 0]]
    return result
