import math

import numpy as np
import pandas as pd


def quantile_crps(quantiles: np.ndarray, levels: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Approximate the CRPS of each forecast by twice the mean pinball loss over its quantiles.

    Parameters
    ----------
    quantiles : np.ndarray
        Forecast quantiles, shape ``(n, n_levels)``.
    levels : np.ndarray
        The quantile levels, shape ``(n_levels,)``, in (0, 1).
    target : np.ndarray
        Realised values, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        CRPS per forecast, shape ``(n,)`` (lower is better).
    """
    error = target[:, None] - quantiles
    pinball = np.maximum(levels * error, (levels - 1) * error)
    return 2 * pinball.mean(axis=1)


def mase(point: np.ndarray, target: np.ndarray, naive_scale: float) -> float:
    """Mean absolute error scaled by the in-sample error of the naive (no change) forecast.

    Parameters
    ----------
    point : np.ndarray
        Point forecasts, shape ``(n,)``.
    target : np.ndarray
        Realised values, shape ``(n,)``.
    naive_scale : float
        Mean absolute error of the naive forecast on the training data (> 0).

    Returns
    -------
    float
        Below 1 beats the naive forecast.

    Raises
    ------
    ValueError
        If `naive_scale` is not positive.
    """
    if naive_scale <= 0:
        raise ValueError(f"naive_scale must be > 0, got {naive_scale}")
    return float(np.mean(np.abs(point - target)) / naive_scale)


def rank_ic(forecast: pd.Series, realised: pd.Series) -> float:
    """Spearman rank correlation between forecasts and realised returns (pairs with NaN dropped).

    NaN, without a warning, when fewer than two pairs remain or either side is constant (the
    correlation is not defined).
    """
    both = pd.concat([forecast, realised], axis=1).dropna()
    if len(both) < 2 or (both.nunique() < 2).any():
        return math.nan
    ranks = both.rank()
    return float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1]))


def hit_rate(forecast: np.ndarray, realised: np.ndarray) -> float:
    """Share of forecasts with the same sign as the realised return (zeros left out)."""
    mask = (forecast != 0) & (realised != 0)
    return float(np.mean(np.sign(forecast[mask]) == np.sign(realised[mask])))


def panel_rank_ic(forecast: pd.DataFrame, realised: pd.DataFrame, *, min_assets: int) -> pd.Series:
    """Rank correlation across assets at each time (the cross-sectional rank IC).

    Parameters
    ----------
    forecast, realised : pd.DataFrame
        Shape ``(n_times, n_assets)``; NaN where missing.
    min_assets : int
        Times with fewer assets having both values are skipped.

    Returns
    -------
    pd.Series
        Rank IC per time; its mean is the panel rank IC.
    """
    both = forecast.notna() & realised.notna()
    ranks_forecast = forecast.where(both).rank(axis=1)
    ranks_realised = realised.where(both).rank(axis=1)
    values = ranks_forecast.corrwith(ranks_realised, axis=1)
    return values[both.sum(axis=1) >= min_assets]
