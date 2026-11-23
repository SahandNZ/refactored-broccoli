from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model

from src.models import QUANTILE_LEVELS
from src.utils import last_at_or_before

NORMAL_QUANTILES = np.array([NormalDist().inv_cdf(level) for level in QUANTILE_LEVELS])
"""The nine standard normal quantiles, for levels 0.1 to 0.9."""

GROUP = ["asset", "horizon"]
"""Every fitted baseline has its own parameters per asset and horizon."""


def _group_rows(frame: pd.DataFrame, keys: str | list[str]) -> dict[object, np.ndarray]:
    # the positions of the rows of each group, e.g. {("BTCUSDT", 1): array([0, 3, ...]), ...}
    return {key: np.asarray(rows) for key, rows in frame.groupby(keys).indices.items()}


def _per_group(test: pd.DataFrame, table: pd.Series | pd.DataFrame) -> np.ndarray:
    # look up each test row's (asset, horizon) value; a group without training data is NaN
    return table.reindex(pd.MultiIndex.from_frame(test[GROUP])).to_numpy(dtype=float)


def random_walk(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast a zero-centred normal with the root mean square of the training labels.

    The price is expected not to move; the spread ``s = sqrt(mean(label ** 2))`` is the
    asset's past label spread around zero, per asset and horizon.

    Parameters
    ----------
    train, test : pd.DataFrame
        Label rows (``asset``, ``horizon``, ``label``), shapes ``(n_train, ...)`` and
        ``(n_test, ...)``.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n_test, 9)``.
    """
    spread = (train["label"] ** 2).groupby([train["asset"], train["horizon"]]).mean() ** 0.5
    return _per_group(test, spread)[:, None] * NORMAL_QUANTILES


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Forecast the empirical quantiles of the training labels, per asset and horizon.

    Parameters
    ----------
    train, test : pd.DataFrame
        Label rows (``asset``, ``horizon``, ``label``).

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n_test, 9)``.
    """
    table = train.groupby(GROUP)["label"].quantile(np.asarray(QUANTILE_LEVELS)).unstack()
    return _per_group(test, table)


def ewma_normal(test: pd.DataFrame) -> np.ndarray:
    """Forecast the standard normal: the label is already divided by the EWMA volatility.

    Parameters
    ----------
    test : pd.DataFrame
        Label rows, shape ``(n_test, ...)``.

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n_test, 9)``, the same for every row.
    """
    return np.tile(NORMAL_QUANTILES, (len(test), 1))


def lag_features(
    frame: pd.DataFrame, closes: dict[str, pd.Series], lags: tuple[int, ...]
) -> np.ndarray:
    """Compute the last vol-scaled log returns before each decision (the linear inputs).

    Lag ``l`` is ``log(P[end] / P[end - l]) / (sigma * sqrt(l))`` with ``end`` the asset's last
    close at or before the decision time: the return over the last ``l`` own bars, the unit
    of the label's horizon.

    Parameters
    ----------
    frame : pd.DataFrame
        Label rows (``time``, ``asset``, ``sigma``), shape ``(n, ...)``.
    closes : dict of str to pd.Series
        1h closes per asset, indexed by close times (UTC).
    lags : tuple of int
        Lags in own bars.

    Returns
    -------
    np.ndarray
        Features, shape ``(n, len(lags))``; NaN where a lag reaches before the first close.
    """
    result = np.full((len(frame), len(lags)), np.nan)
    sigma = frame["sigma"].to_numpy(dtype=float)
    for asset, rows in _group_rows(frame, "asset").items():
        series = closes[str(asset)]
        log_price = np.log(series.to_numpy(dtype=float))
        ends = last_at_or_before(
            pd.DatetimeIndex(series.index), pd.DatetimeIndex(frame["time"].iloc[rows])
        )
        for column, lag in enumerate(lags):
            before = ends - lag
            change = np.where(
                before >= 0, log_price[ends] - log_price[np.maximum(before, 0)], np.nan
            )
            result[rows, column] = change / (sigma[rows] * np.sqrt(lag))
    return result


def linear(
    train: pd.DataFrame,
    test: pd.DataFrame,
    closes: dict[str, pd.Series],
    *,
    lags: tuple[int, ...],
    ridge: float,
) -> np.ndarray:
    """Forecast a ridge regression on the lags plus the quantiles of its training residuals.

    Per asset and horizon, ``w = argmin |y - X w|^2 + ridge |w|^2`` (no intercept) on the
    training rows with every lag known (`lag_features`); the forecast is ``x w`` plus the
    empirical quantiles of the in-sample residuals ``y - X w``.

    Parameters
    ----------
    train, test : pd.DataFrame
        Label rows (``time``, ``asset``, ``horizon``, ``label``, ``sigma``).
    closes : dict of str to pd.Series
        1h closes per asset, indexed by close times (UTC).
    lags : tuple of int
        Lags in own bars.
    ridge : float
        L2 penalty (> 0).

    Returns
    -------
    np.ndarray
        Label quantiles, shape ``(n_test, 9)``; NaN for a group without training rows.
    """
    train_x, test_x = lag_features(train, closes, lags), lag_features(test, closes, lags)
    labels = np.asarray(train["label"], dtype=float)
    test_groups = _group_rows(test, GROUP)
    result = np.full((len(test), len(QUANTILE_LEVELS)), np.nan)
    for key, rows in _group_rows(train, GROUP).items():
        known = rows[np.isfinite(train_x[rows]).all(axis=1)]
        if key not in test_groups or len(known) == 0:
            continue
        x, y = train_x[known], labels[known]
        weights = np.linalg.solve(x.T @ x + ridge * np.eye(len(lags)), x.T @ y)
        spread = np.quantile(y - x @ weights, QUANTILE_LEVELS)
        test_rows = test_groups[key]
        result[test_rows] = (test_x[test_rows] @ weights)[:, None] + spread
    return result


def _garch_variance(
    returns: pd.Series, start: pd.Timestamp, horizon: int
) -> tuple[pd.DataFrame | None, int]:
    # Fit GARCH(1,1) on the returns closing by `start`, then filter every later return with the
    # fitted parameters. Returns are divided by their training standard deviation first, the
    # scaling arch asks for. Row k: variance of the sum of the next 1 .. horizon returns. Also
    # arch's convergence flag: 0, or the flag with no variance when the fit does not converge
    train = returns[returns.index <= start]
    scale = float(train.std())
    fitted = arch_model(train / scale, mean="zero", vol="garch", p=1, q=1, dist="normal").fit(
        disp="off"
    )
    if fitted.convergence_flag != 0:
        return None, int(fitted.convergence_flag)
    model = arch_model(returns / scale, mean="zero", vol="garch", p=1, q=1, dist="normal")
    forecast = model.fix(fitted.params).forecast(
        horizon=horizon, start=len(train) - 1, reindex=False
    )
    return forecast.variance.cumsum(axis=1) * scale**2, 0


def garch_normal(
    test: pd.DataFrame, closes: dict[str, pd.Series], *, start: pd.Timestamp, train_days: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Forecast a zero-centred normal with the h-step variance of a GARCH(1,1).

    Per asset, GARCH(1,1) with normal errors and zero mean is fitted (arch) on the hourly log
    returns of the `train_days` before `start`; its parameters stay fixed through the window.
    At a decision ``t`` whose last return ``r_k`` closed at or before ``t``, the variance of
    the next ``h`` own-bar returns (the label's horizon) is ``V_h = sum_{j=1..h}
    E[sigma^2_{k+j}]``, so the label quantiles are ``z * sqrt(V_h) / (sigma * sqrt(h))``.
    An asset whose fit does not converge gets no forecast (NaN rows) and is returned with
    arch's convergence flag, so the caller leaves it out and lists it.

    Parameters
    ----------
    test : pd.DataFrame
        Label rows (``time``, ``asset``, ``horizon``, ``sigma``), all of one window.
    closes : dict of str to pd.Series
        1h closes per asset, indexed by close times (UTC).
    start : pd.Timestamp
        The window start: the fit uses only returns that closed by then.
    train_days : int
        Days of returns in the fit.

    Returns
    -------
    tuple of (np.ndarray, dict of str to int)
        Label quantiles, shape ``(n_test, 9)``, NaN for an asset not fitted; and the assets
        whose fit did not converge, with arch's convergence flag.

    References
    ----------
    T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity", Journal of
    Econometrics 31(3), 1986.
    """
    result = np.full((len(test), len(QUANTILE_LEVELS)), np.nan)
    not_fitted: dict[str, int] = {}
    horizons = test["horizon"].to_numpy()
    sigma = test["sigma"].to_numpy(dtype=float)
    for asset, rows in _group_rows(test, "asset").items():
        series = closes[str(asset)]
        log_closes = np.log(series.to_numpy(dtype=float))
        returns = pd.Series(np.diff(log_closes), index=series.index[1:], name=str(asset))
        returns = returns[returns.index > start - pd.Timedelta(days=train_days)]
        variance, flag = _garch_variance(returns, start, int(horizons[rows].max()))
        if variance is None:
            not_fitted[str(asset)] = flag
            continue
        times = pd.DatetimeIndex(test["time"].iloc[rows])
        origins = last_at_or_before(pd.DatetimeIndex(variance.index), times)
        spread = np.sqrt(variance.to_numpy()[origins, horizons[rows] - 1])
        label_spread = spread / (sigma[rows] * np.sqrt(horizons[rows]))
        result[rows] = label_spread[:, None] * NORMAL_QUANTILES
    return result, not_fitted
