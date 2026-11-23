"""Q1: the fitted baselines of 02-forecasts refitted on the last `window` own bars only.

Each asset's horizon-1 decisions of the development and test folds are one per own bar, in
order (02-forecasts: one decision per own close), so the `window` decisions before a decision
are its last `window` own bars, and their labels all closed by the decision time. A baseline
refitted on them sees what a zero-shot model reads: the last 512 bars. A decision with fewer
than `window` earlier decisions gets no forecast (the first months of d1 for markets that
close). The folds are contiguous, but at each fold's end the cache drops the decision whose
label reaches into the next fold, so a window may hold such a gap (one bar's return missing):
a ridge lag that spans a gap is left out (NaN), and a GARCH fit or filter simply lacks that
return; the random walk and climatology lose one label from the window.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
from numpy.lib.stride_tricks import sliding_window_view

from src.benchmark import QUANTILE_COLUMNS, ForecastCache
from src.models import QUANTILE_LEVELS

NORMAL_QUANTILES = np.array([NormalDist().inv_cdf(level) for level in QUANTILE_LEVELS])
"""The nine standard normal quantiles, for levels 0.1 to 0.9."""

CONTEXT_BASELINES = ("random-walk", "climatology", "garch-normal", "linear")
"""The baselines of 02-forecasts that fit parameters, refitted here on the context."""

KEYS = ["time", "asset", "horizon"]
"""The columns that name one decision (the cache's join keys)."""


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """How the context baselines are fitted.

    Attributes
    ----------
    window : int
        Own bars in the fit (the zero-shot models' ``lookback``).
    lags : tuple of int
        The ridge regression's lags in own bars (02-forecasts' ``linear_lags``).
    ridge : float
        The ridge penalty (02-forecasts' ``linear_ridge``).
    workers : int
        Processes of the GARCH fits.
    """

    window: int
    lags: tuple[int, ...]
    ridge: float
    workers: int


def rolling_random_walk(labels: np.ndarray, window: int) -> np.ndarray:
    """Return the random walk's label quantiles from the previous `window` labels.

    The spread is ``sqrt(mean(label ** 2))`` over labels ``i - window .. i - 1``, as
    02-forecasts' `random_walk` over its training labels.

    Parameters
    ----------
    labels : np.ndarray
        One asset's horizon-1 labels in decision order, shape ``(n,)``.
    window : int
        Labels in the fit.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n, 9)``; NaN rows for the first `window` decisions.
    """
    result = np.full((len(labels), len(QUANTILE_LEVELS)), np.nan)
    if len(labels) <= window:
        return result
    squares = np.concatenate([[0.0], np.cumsum(labels**2)])
    mean = (squares[window:-1] - squares[: -window - 1]) / window
    result[window:] = np.sqrt(mean)[:, None] * NORMAL_QUANTILES
    return result


def rolling_climatology(labels: np.ndarray, window: int) -> np.ndarray:
    """Return climatology's label quantiles: the deciles of the previous `window` labels.

    Linear interpolation, as 02-forecasts' `climatology` (pandas' quantile).

    Parameters
    ----------
    labels : np.ndarray
        One asset's horizon-1 labels in decision order, shape ``(n,)``.
    window : int
        Labels in the fit.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n, 9)``; NaN rows for the first `window` decisions.
    """
    result = np.full((len(labels), len(QUANTILE_LEVELS)), np.nan)
    if len(labels) <= window:
        return result
    windows = sliding_window_view(labels[:-1], window)
    result[window:] = np.quantile(windows, QUANTILE_LEVELS, axis=1).T
    return result


def _lags(
    log_returns: np.ndarray, sigma: np.ndarray, gaps: np.ndarray, lags: tuple[int, ...]
) -> np.ndarray:
    # lag l at decision i: the log return over the last l own bars, sum of the horizon-1 log
    # returns of decisions i - l .. i - 1, divided by sigma_i * sqrt(l) (02-forecasts'
    # `lag_features`); NaN where a lag reaches before the first decision or across a gap
    # (gaps[k]: decision k does not follow decision k - 1 by one own bar)
    sums = np.concatenate([[0.0], np.cumsum(log_returns)])
    breaks = np.concatenate([[0], np.cumsum(gaps.astype(int))])
    features = np.full((len(log_returns), len(lags)), np.nan)
    for column, lag in enumerate(lags):
        rows = np.arange(lag, len(log_returns))
        values = (sums[rows] - sums[rows - lag]) / (sigma[rows] * np.sqrt(lag))
        crosses = breaks[rows + 1] - breaks[rows - lag + 1] > 0
        features[rows, column] = np.where(crosses, np.nan, values)
    return features


def rolling_ridge(
    labels: np.ndarray,
    log_returns: np.ndarray,
    sigma: np.ndarray,
    gaps: np.ndarray,
    settings: ContextSettings,
) -> np.ndarray:
    """Return the ridge regression's label quantiles, refitted on the previous `window` decisions.

    At decision ``i`` the weights solve ``(X'X + ridge I) w = X'y`` over the decisions
    ``i - window .. i - 1`` with every lag known, and the forecast is ``x_i w`` plus the
    deciles of those decisions' residuals, as 02-forecasts' `linear` over its training labels.

    Parameters
    ----------
    labels, log_returns, sigma : np.ndarray
        One asset's horizon-1 labels, log returns and EWMA volatilities in decision order,
        each of shape ``(n,)``.
    gaps : np.ndarray
        Booleans, shape ``(n,)``: decision ``k`` does not follow decision ``k - 1`` by one own
        bar (a decision the cache left out between them); a lag across a gap is NaN.
    settings : ContextSettings
        The window, lags and penalty.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n, 9)``; NaN rows for the first `window` decisions and for
        a decision without every lag.
    """
    window = settings.window
    features = _lags(log_returns, sigma, gaps, settings.lags)
    result = np.full((len(labels), len(QUANTILE_LEVELS)), np.nan)
    penalty = settings.ridge * np.eye(len(settings.lags))
    for i in range(window, len(labels)):
        x, y = features[i - window : i], labels[i - window : i]
        known = np.isfinite(x).all(axis=1)
        if not known.any() or not np.isfinite(features[i]).all():
            continue
        x, y = x[known], y[known]
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        result[i] = features[i] @ weights + np.quantile(y - x @ weights, QUANTILE_LEVELS)
    return result


def _garch_day(task: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
    # one daily refit on `series`: its first `window` returns fit GARCH(1,1) (zero mean, normal
    # errors, returns divided by their standard deviation, as 02-forecasts); the fitted
    # parameters then filter the rest, and row k is the one-step variance from origin
    # window - 1 + k, the variance of the return after the day's k-th decision
    series, window = task
    scale = float(series[:window].std())
    if not scale > 0:
        return np.full(len(series) - window + 1, np.nan), -1
    # arch warns when the optimizer fails; the flag below records it, so the warning is
    # not repeated on stderr thousands of times
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = arch_model(
            series[:window] / scale, mean="zero", vol="garch", p=1, q=1, dist="normal"
        ).fit(disp="off")
    if fitted.convergence_flag != 0:
        return np.full(len(series) - window + 1, np.nan), int(fitted.convergence_flag)
    model = arch_model(series / scale, mean="zero", vol="garch", p=1, q=1, dist="normal")
    forecast = model.fix(fitted.params).forecast(horizon=1, start=window - 1, reindex=False)
    return forecast.variance.to_numpy()[:, 0] * scale**2, 0


def garch_spreads(
    log_returns: np.ndarray, days: np.ndarray, settings: ContextSettings, pool: ProcessPoolExecutor
) -> tuple[np.ndarray, int]:
    """Return GARCH(1,1)'s one-step standard deviation at each decision, refitted daily.

    On the first decision of each day the model is fitted on the previous `window` returns
    (the horizon-1 log returns of the previous decisions); every decision of that day then gets
    the one-step variance filtered with those parameters through the day's earlier returns.

    Parameters
    ----------
    log_returns : np.ndarray
        One asset's horizon-1 log returns in decision order, shape ``(n,)``; return ``i`` is
        the move after decision ``i``.
    days : np.ndarray
        The UTC day of each decision, shape ``(n,)``, in order.
    settings : ContextSettings
        The window.
    pool : ProcessPoolExecutor
        The worker processes of the daily fits.

    Returns
    -------
    tuple of (np.ndarray, int)
        The standard deviations, shape ``(n,)`` (NaN for the first `window` decisions and on
        a day whose fit did not converge), and how many daily fits did not converge.
    """
    window = settings.window
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    stops = np.r_[starts[1:], len(days)]
    tasks: list[tuple[np.ndarray, int]] = []
    spans: list[tuple[int, int]] = []
    for start, stop in zip(starts, stops, strict=True):
        first = max(int(start), window)
        if first >= stop:
            continue
        tasks.append((log_returns[first - window : stop - 1], window))
        spans.append((first, int(stop)))
    spreads = np.full(len(log_returns), np.nan)
    failed = 0
    for (first, stop), (variance, flag) in zip(
        spans, pool.map(_garch_day, tasks, chunksize=32), strict=True
    ):
        failed += flag != 0
        spreads[first:stop] = np.sqrt(variance[: stop - first])
    return spreads, failed


def context_forecasts(
    cache: ForecastCache, *, windows: list[str], settings: ContextSettings
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return the forecasts of every context baseline at horizon 1 and the GARCH fits that failed.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache (its labels of `windows` at horizon 1).
    windows : list of str
        The contiguous folds whose decisions are forecast (d1-d4 and t1-t4).
    settings : ContextSettings
        The window, lags, penalty and workers.

    Returns
    -------
    tuple of (dict of str to pd.DataFrame, pd.DataFrame)
        For each name of `CONTEXT_BASELINES`, its forecast rows (``time``, ``asset``,
        ``horizon``, ``seed`` 0 and `QUANTILE_COLUMNS`; only decisions with a full window and
        a valid fit); and per asset the GARCH daily fits that did not converge
        (``asset``, ``failed``).
    """
    labels = cache.labels[(cache.labels["horizon"] == 1) & cache.labels["window"].isin(windows)]
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in CONTEXT_BASELINES}
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        for asset, rows in labels.sort_values(["asset", "time"]).groupby("asset", sort=True):
            label = rows["label"].to_numpy(dtype=float)
            log_returns = rows["log_return"].to_numpy(dtype=float)
            sigma = rows["sigma"].to_numpy(dtype=float)
            days = rows["time"].dt.floor("D").to_numpy()
            follows = (rows["label_time"].shift(1) == rows["time"]).to_numpy()
            gaps = ~follows
            gaps[0] = False
            spreads, failed = garch_spreads(log_returns, days, settings, pool)
            failures.append({"asset": asset, "failed": failed})
            quantiles = {
                "random-walk": rolling_random_walk(label, settings.window),
                "climatology": rolling_climatology(label, settings.window),
                "linear": rolling_ridge(label, log_returns, sigma, gaps, settings),
                "garch-normal": (spreads / sigma)[:, None] * NORMAL_QUANTILES,
            }
            keys = rows[KEYS].reset_index(drop=True).assign(seed=0)
            for name, values in quantiles.items():
                known = np.isfinite(values).all(axis=1)
                frame = pd.concat(
                    [
                        keys[known].reset_index(drop=True),
                        pd.DataFrame(values[known], columns=list(QUANTILE_COLUMNS)),
                    ],
                    axis=1,
                )
                parts[name].append(frame)
    forecasts = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    return forecasts, pd.DataFrame(failures, columns=["asset", "failed"])


@dataclass(frozen=True, slots=True)
class ContextCache(ForecastCache):
    """A forecast cache whose fitted baselines are the context baselines, at horizon 1 only.

    Attributes
    ----------
    context : dict of str to pd.DataFrame
        The forecast rows of each context baseline (`context_forecasts`); every other model
        reads its forecasts from the cache.
    """

    context: dict[str, pd.DataFrame]

    def forecasts(self, model: str) -> pd.DataFrame:
        """Return `model`'s labelled forecasts: a context baseline's, or the cache's.

        Parameters
        ----------
        model : str
            A model of the cache.

        Returns
        -------
        pd.DataFrame
            The label columns, ``seed`` and `QUANTILE_COLUMNS`, one row per decision and seed.
        """
        if model in self.context:
            return self.labels.merge(self.context[model], on=KEYS, validate="one_to_many")
        frame = ForecastCache.forecasts(self, model)
        return frame[frame["horizon"] == 1].reset_index(drop=True)


def context_cache(
    cache: ForecastCache, context: dict[str, pd.DataFrame], windows: list[str]
) -> ContextCache:
    """Return `cache` at horizon 1 in `windows`, with the context baselines in place of its own.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    context : dict of str to pd.DataFrame
        The forecasts of `context_forecasts`.
    windows : list of str
        The windows kept (the development and test folds).

    Returns
    -------
    ContextCache
        The cache whose labels are those of horizon 1 in `windows`.
    """
    labels = cache.labels[(cache.labels["horizon"] == 1) & cache.labels["window"].isin(windows)]
    base = replace(cache, labels=labels.reset_index(drop=True))
    return ContextCache(base.folder, base.labels, base.models, base.assets, context)
