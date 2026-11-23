"""Significance tests.

Paired t over (fold, seed) pairs, the Diebold-Mariano test of equal forecast accuracy, least
squares with Newey-West errors, Holm's correction and block-bootstrap intervals and tests.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
from arch.bootstrap import optimal_block_length
from scipy.special import stdtr, stdtrit


def paired_t(differences: np.ndarray) -> dict[str, float]:
    """Paired t-test of the mean of differences against zero.

    Parameters
    ----------
    differences : np.ndarray
        One difference per pair (e.g. per fold and seed), shape ``(n,)``, n >= 2.

    Returns
    -------
    dict
        ``mean``, ``t``, ``n`` and ``wins`` (count of positive differences).

    Raises
    ------
    ValueError
        If there are fewer than two pairs.
    """
    if len(differences) < 2:
        raise ValueError(f"need at least 2 pairs, got {len(differences)}")
    mean = float(np.mean(differences))
    error = float(np.std(differences, ddof=1)) / math.sqrt(len(differences))
    return {
        "mean": mean,
        "t": mean / error if error > 0 else math.inf * np.sign(mean),
        "n": len(differences),
        "wins": int((differences > 0).sum()),
    }


def andrews_lags(values: np.ndarray) -> int:
    """Return Andrews' automatic number of Newey-West lags for a series (Bartlett kernel).

    The AR(1) plug-in rule: with ``rho`` the first-order autocorrelation of the centred
    series, ``a = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)`` and the bandwidth
    ``1.1447 (a n)^(1/3)``; the lags are its integer part, at most ``n - 1``. A series
    without variance has 0 lags.

    Parameters
    ----------
    values : np.ndarray
        The series in time order, shape ``(n,)``.

    Returns
    -------
    int
        The number of autocovariance lags, ``0 <= lags <= n - 1``.

    References
    ----------
    D. Andrews, "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
    Estimation", Econometrica 59(3), 1991.
    """
    centred = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centred)
    variance = float(centred @ centred)
    if n < 2 or variance == 0:
        return 0
    rho = float(centred[1:] @ centred[:-1]) / variance
    rho = min(max(rho, -0.99), 0.99)
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return min(int(1.1447 * (alpha * n) ** (1 / 3)), n - 1)


def newey_west_variance(centred: np.ndarray, *, lags: int) -> np.ndarray:
    """Return the Newey-West long-run variance of each row of centred series.

    ``gamma_0 + 2 sum_{j=1..lags} (1 - j / (lags + 1)) gamma_j``, ``gamma_j`` the
    autocovariance at lag ``j`` (divided by ``n``), Bartlett weights.

    Parameters
    ----------
    centred : np.ndarray
        Series with their mean taken out, shape ``(n_series, n)``.
    lags : int
        Autocovariance lags (``0 <= lags < n``).

    Returns
    -------
    np.ndarray
        One variance per series, shape ``(n_series,)``.

    References
    ----------
    W. Newey and K. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and
    Autocorrelation Consistent Covariance Matrix", Econometrica 55(3), 1987.
    """
    n = centred.shape[1]
    variance = (centred * centred).sum(axis=1) / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance = variance + 2 * weight * (centred[:, lag:] * centred[:, :-lag]).sum(axis=1) / n
    return variance


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, overlap: int) -> dict[str, float]:
    """Diebold-Mariano test that two forecasts have equal expected loss.

    The loss difference ``d = loss_a - loss_b`` overlaps: each value shares its label with
    the next `overlap` ones (``h - 1`` for ``h``-step forecasts made every step). Its
    Newey-West variance takes ``max(overlap, andrews_lags(d))`` lags (Bartlett weights), the
    statistic gets the Harvey, Leybourne and Newbold small-sample correction for
    ``h = overlap + 1`` and is read against Student's t with ``n - 1`` degrees of freedom.
    A difference without variance (`is_constant`) has no test (NaN).

    Parameters
    ----------
    loss_a, loss_b : np.ndarray
        Losses of the two forecasts on the same targets, in time order, shape ``(n,)``.
    overlap : int
        How many later values share a label with each value (>= 0).

    Returns
    -------
    dict
        ``statistic`` (negative when `loss_a` is smaller, i.e. a is better), ``p_value``
        (two-sided), ``p_less`` (one-sided, H1: a has the smaller expected loss), ``lags``
        (the Newey-West lags used) and ``mean_difference``.

    Raises
    ------
    ValueError
        If `overlap` is negative or the series have fewer than two values.

    References
    ----------
    F. Diebold and R. Mariano, "Comparing Predictive Accuracy", Journal of Business and
    Economic Statistics 13(3), 1995; D. Harvey, S. Leybourne and P. Newbold, "Testing the
    Equality of Prediction Mean Squared Errors", International Journal of Forecasting 13(2),
    1997.
    """
    difference = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(difference)
    if overlap < 0 or n < 2:
        raise ValueError(f"need overlap >= 0 and at least 2 values, got {overlap}, {n}")
    scale = float(np.mean(np.abs(loss_a)) + np.mean(np.abs(loss_b)))
    if is_constant(difference, scale=scale):
        nan = math.nan
        mean = float(difference.mean())
        return {"statistic": nan, "p_value": nan, "p_less": nan, "lags": 0, "mean_difference": mean}
    lags = min(max(overlap, andrews_lags(difference)), n - 1)
    centred = (difference - difference.mean())[None, :]
    variance = float(newey_west_variance(centred, lags=lags)[0])
    statistic = dm_statistic(float(difference.mean()), variance, n=n, overlap=overlap)
    return {
        "statistic": statistic,
        "p_value": float(2 * stdtr(n - 1, -abs(statistic))),
        "p_less": float(stdtr(n - 1, statistic)),
        "lags": lags,
        "mean_difference": float(difference.mean()),
    }


CONSTANT_TOLERANCE = 1e-10
"""A series whose standard deviation is at most this part of its scale counts as constant."""


def is_constant(values: np.ndarray, *, scale: float) -> bool:
    """Return whether a series has no variance beyond rounding: std <= tolerance x `scale`.

    A loss difference of two forecasts that differ by a constant is computed with rounding
    noise of about 1e-16 of the losses; that noise must not make a test "significant".

    Parameters
    ----------
    values : np.ndarray
        The series, shape ``(n,)``.
    scale : float
        The size of the numbers it was computed from (e.g. the mean absolute losses).

    Returns
    -------
    bool
        True when ``std(values) <= CONSTANT_TOLERANCE * scale``.
    """
    return float(np.std(values)) <= CONSTANT_TOLERANCE * scale


def dm_statistic(mean: float, variance: float, *, n: int, overlap: int) -> float:
    """Return the corrected Diebold-Mariano statistic from a mean and a long-run variance.

    ``mean / sqrt(variance / n)`` times the Harvey-Leybourne-Newbold factor
    ``sqrt((n + 1 - 2h + h (h - 1) / n) / n)`` with ``h = overlap + 1``; NaN when the
    variance is not positive (or the factor is not defined), so a constant difference is
    never "significant".

    Parameters
    ----------
    mean : float
        The mean loss difference.
    variance : float
        Its Newey-West long-run variance (`newey_west_variance`).
    n : int
        The number of values.
    overlap : int
        As in `diebold_mariano`.

    Returns
    -------
    float
        The statistic, or NaN.
    """
    horizon = overlap + 1
    factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if not variance > 0 or not factor > 0:
        return math.nan
    return mean / math.sqrt(variance / n) * math.sqrt(factor)


@dataclass(frozen=True, slots=True)
class Regression:
    """A least-squares fit with Newey-West (HAC) t statistics.

    Attributes
    ----------
    coefficients : np.ndarray
        One per column of the design matrix, shape ``(k,)``.
    t : np.ndarray
        The t statistic of each coefficient, shape ``(k,)``.
    r2 : float
        The share of the variance of the response the fit explains.
    """

    coefficients: np.ndarray
    t: np.ndarray
    r2: float


def newey_west_ols(response: np.ndarray, design: np.ndarray, *, lags: int) -> Regression:
    """Fit least squares and test each coefficient with Newey-West (HAC) standard errors.

    The covariance is the sandwich ``(X'X)^-1 S (X'X)^-1``: ``S`` sums the outer products of
    the scores ``x_t e_t`` and their autocovariances up to `lags`, with Bartlett weights
    ``1 - lag / (lags + 1)``, times the small-sample factor ``n / (n - k)`` (statsmodels'
    ``cov_type="HAC"`` with ``use_correction=True``).

    Parameters
    ----------
    response : np.ndarray
        Shape ``(n,)``.
    design : np.ndarray
        Shape ``(n, k)``; include a column of ones for an intercept.
    lags : int
        Autocovariance lags of the scores (>= 0).

    Returns
    -------
    Regression
        Coefficients, t statistics and R^2.

    Raises
    ------
    ValueError
        If there are not more rows than columns or `lags` is negative.

    References
    ----------
    W. Newey and K. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and
    Autocorrelation Consistent Covariance Matrix", Econometrica 55(3), 1987.
    """
    n, k = design.shape
    if n <= k or lags < 0:
        raise ValueError(f"need more rows than columns and lags >= 0, got {n} x {k}, lags {lags}")
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ response
    residuals = response - design @ coefficients
    scores = design * residuals[:, None]
    middle = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        middle += (1 - lag / (lags + 1)) * (cross + cross.T)
    covariance = inverse @ middle @ inverse * n / (n - k)
    centred = response - response.mean()
    r2 = 1 - float(residuals @ residuals) / float(centred @ centred)
    return Regression(coefficients, coefficients / np.sqrt(np.diag(covariance)), r2)


def verdict(differences: np.ndarray, *, min_pairs: int = 3, t_threshold: float = 2.0) -> str:
    """Turn paired differences (candidate minus reference; positive is better) into a claim.

    A difference is claimed only when it is significant; a gap inside the seed noise is
    "no difference".

    Parameters
    ----------
    differences : np.ndarray
        One difference per (fold, seed) pair, shape ``(n,)``.
    min_pairs : int, optional
        Fewest pairs needed for any claim (three seeds at least).
    t_threshold : float, optional
        The paired t above which a difference is claimed.

    Returns
    -------
    str
        ``"better"``, ``"worse"``, ``"no difference"`` or ``"too few runs"``.
    """
    if len(differences) < min_pairs:
        return "too few runs"
    result = paired_t(differences)
    if result["t"] > t_threshold:
        return "better"
    if result["t"] < -t_threshold:
        return "worse"
    return "no difference"


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2))


def _normal_quantile(probability: float) -> float:
    # bisection on the normal CDF: exact enough for the Deflated Sharpe benchmark
    low, high = -10.0, 10.0
    for _ in range(100):
        middle = (low + high) / 2
        if _normal_cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def deflated_sharpe_ratio(returns: np.ndarray, *, trial_sharpes: np.ndarray) -> float:
    """Probability that the true Sharpe ratio is above what the best of many trials gets by luck.

    Parameters
    ----------
    returns : np.ndarray
        The chosen strategy's per-period returns, shape ``(n,)``.
    trial_sharpes : np.ndarray
        Per-period (not annualised) Sharpe ratios of every trial that was tried, shape
        ``(n_trials,)``, n_trials >= 2.

    Returns
    -------
    float
        DSR in [0, 1]; above 0.95 is significant at 5 %.

    References
    ----------
    D. Bailey, M. López de Prado, "The Deflated Sharpe Ratio", Journal of Portfolio Management
    40(5), 2014.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    sharpe = returns.mean() / returns.std(ddof=1)
    centred = returns - returns.mean()
    skew = float((centred**3).mean() / centred.std() ** 3)
    kurtosis = float((centred**4).mean() / centred.std() ** 4)
    trials = len(trial_sharpes)
    euler = 0.5772156649
    spread = float(np.std(trial_sharpes, ddof=1))
    benchmark = spread * (
        (1 - euler) * _normal_quantile(1 - 1 / trials)
        + euler * _normal_quantile(1 - 1 / (trials * math.e))
    )
    denominator = math.sqrt(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2)
    return _normal_cdf((sharpe - benchmark) * math.sqrt(n - 1) / denominator)


def probability_of_backtest_overfitting(returns: np.ndarray, *, n_blocks: int) -> float:
    """Probability that the configuration best in-sample is below the median out of sample.

    Combinatorially symmetric cross-validation: the time axis is cut into `n_blocks` blocks;
    every half of the blocks serves once as in-sample, the rest as out-of-sample.

    Parameters
    ----------
    returns : np.ndarray
        Per-period returns of every configuration tried, shape ``(n_periods, n_configurations)``.
    n_blocks : int
        Even number of blocks (e.g. 16).

    Returns
    -------
    float
        PBO in [0, 1]; near 0.5 or above means the selection is not better than chance.

    References
    ----------
    D. Bailey, J. Borwein, M. López de Prado, Q. Zhu, "The Probability of Backtest
    Overfitting", Journal of Computational Finance 20(4), 2017.
    """
    if n_blocks % 2:
        raise ValueError(f"n_blocks must be even, got {n_blocks}")
    blocks = np.array_split(np.asarray(returns, dtype=float), n_blocks)

    def sharpe(chosen: tuple[int, ...]) -> np.ndarray:
        stacked = np.concatenate([blocks[i] for i in chosen])
        deviation = stacked.std(axis=0, ddof=1)
        return np.divide(
            stacked.mean(axis=0), deviation, out=np.zeros_like(deviation), where=deviation > 0
        )

    below: list[bool] = []
    for inside in itertools.combinations(range(n_blocks), n_blocks // 2):
        outside = tuple(i for i in range(n_blocks) if i not in inside)
        best = int(np.argmax(sharpe(inside)))
        out_of_sample = sharpe(outside)
        rank = (out_of_sample < out_of_sample[best]).sum() / (len(out_of_sample) - 1)
        below.append(rank < 0.5)
    return float(np.mean(below))


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for several tests of one question.

    A NaN p-value marks a cell without a test: it stays NaN and does not count in the number
    of tests ``m``.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values, shape ``(n,)``; NaN where there is no test.

    Returns
    -------
    np.ndarray
        Adjusted p-values, same order, each at most 1; NaN where the input is NaN.

    References
    ----------
    S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scandinavian Journal of
    Statistics 6(2), 1979.
    """
    p_values = np.asarray(p_values, dtype=float)
    is_tested = ~np.isnan(p_values)
    tested = p_values[is_tested]
    order = np.argsort(tested)
    count = len(tested)
    adjusted_sorted = np.maximum.accumulate((count - np.arange(count)) * tested[order])
    adjusted_tested = np.empty(count)
    adjusted_tested[order] = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.full(len(p_values), np.nan)
    adjusted[is_tested] = adjusted_tested
    return adjusted


def block_indices(
    length: int, rng: np.random.Generator, *, n_resamples: int, block: int
) -> np.ndarray:
    """Draw the positions of moving-block bootstrap resamples of a series.

    Each resample glues random runs of `block` consecutive positions until it is as long as
    the series (the last run is cut), so it keeps the series' autocorrelation within a block.

    Parameters
    ----------
    length : int
        The length of the series.
    rng : np.random.Generator
        The source of the draws.
    n_resamples : int
        Bootstrap draws.
    block : int
        Block length in positions (1 <= block <= length).

    Returns
    -------
    np.ndarray
        Positions into the series, shape ``(n_resamples, length)``.

    Raises
    ------
    ValueError
        If `block` is below 1 or longer than the series.

    References
    ----------
    H. Künsch, "The Jackknife and the Bootstrap for General Stationary Observations", Annals of
    Statistics 17(3), 1989.
    """
    if not 1 <= block <= length:
        raise ValueError(f"block must be between 1 and the {length} values, got {block}")
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, length - block + 1, size=(n_resamples, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_resamples, -1)[:, :length]


MIN_BLOCK_VALUES = 11
"""The shortest series `arch.bootstrap.optimal_block_length` estimates a block for."""


def politis_white_block(values: np.ndarray) -> int:
    """Return the Politis-White block length of a series for the circular-block bootstrap.

    The automatic choice of `arch.bootstrap.optimal_block_length` (circular block, with the
    correction of Patton, Politis and White 2009), rounded up; 1 for a series without
    variance or shorter than `MIN_BLOCK_VALUES` (the estimator needs that many).

    Parameters
    ----------
    values : np.ndarray
        The series in time order, shape ``(n,)``.

    Returns
    -------
    int
        The block length, at least 1.

    References
    ----------
    D. Politis and H. White, "Automatic Block-Length Selection for the Dependent Bootstrap",
    Econometric Reviews 23(1), 2004.
    """
    if len(values) < MIN_BLOCK_VALUES or np.ptp(values) == 0:
        return 1
    length = float(optimal_block_length(np.asarray(values, dtype=float))["circular"].iloc[0])
    return max(1, math.ceil(length)) if math.isfinite(length) else 1


MIN_BOOTSTRAP_BLOCKS = 5
"""The fewest blocks a block-bootstrap test needs (owner's decision, 2026-11-17)."""

BLOCK_OVERLAP_FLOOR = 4
"""A bootstrap block is at least this many times the label overlap (owner's decision,
2026-11-17, for every study; it was 2): the overlap stays well inside a block."""


def has_enough_blocks(length: int, block: int) -> bool:
    """Return whether a series of `length` values holds at least `MIN_BOOTSTRAP_BLOCKS` blocks.

    A series with fewer blocks is reported as untested: its bootstrap has too few independent
    pieces for a test or an interval.

    Parameters
    ----------
    length : int
        The number of values (e.g. decision times).
    block : int
        The block length in values.

    Returns
    -------
    bool
        ``length >= MIN_BOOTSTRAP_BLOCKS * block``.
    """
    return length >= MIN_BOOTSTRAP_BLOCKS * block


def circular_block_indices(
    length: int, rng: np.random.Generator, *, n_resamples: int, block: int
) -> np.ndarray:
    """Draw the positions of circular-block bootstrap resamples of a series.

    Like `block_indices`, but a block may start at any position and wraps around the end of
    the series, so every value is drawn equally often (no edge effect).

    Parameters
    ----------
    length : int
        The length of the series.
    rng : np.random.Generator
        The source of the draws.
    n_resamples : int
        Bootstrap draws.
    block : int
        Block length in positions (1 <= block <= length).

    Returns
    -------
    np.ndarray
        Positions into the series, shape ``(n_resamples, length)``.

    Raises
    ------
    ValueError
        If `block` is below 1 or longer than the series.

    References
    ----------
    D. Politis and J. Romano, "A Circular Block-Resampling Procedure for Stationary Data", in
    Exploring the Limits of Bootstrap, Wiley, 1992.
    """
    if not 1 <= block <= length:
        raise ValueError(f"block must be between 1 and the {length} values, got {block}")
    n_blocks = math.ceil(length / block)
    starts = rng.integers(0, length, size=(n_resamples, n_blocks))
    positions = (starts[:, :, None] + np.arange(block)) % length
    return positions.reshape(n_resamples, -1)[:, :length]


def _circular_error(
    values: np.ndarray, rng: np.random.Generator, *, n_resamples: int, block: int
) -> tuple[float, int]:
    # the circular-block bootstrap standard error of the mean, and its degrees of freedom
    # (the number of blocks less one)
    values = np.asarray(values, dtype=float)
    if len(values) < 2 * block:
        raise ValueError(f"need at least 2 blocks of {block} values, got {len(values)} values")
    indices = circular_block_indices(len(values), rng, n_resamples=n_resamples, block=block)
    return float(values[indices].mean(axis=1).std(ddof=1)), len(values) // block - 1


def _t_test(estimate: float, error: float, *, df: float, level: float) -> dict[str, float]:
    # a t test of a zero estimate (estimate / bootstrap standard error) and its interval,
    # against Student's t with `df`;
    # without spread the estimate is either exactly zero or certain
    p_value = 1.0 if estimate == 0 else 0.0
    p_greater = 0.0 if estimate > 0 else 1.0
    if error > 0:
        p_value = float(2 * stdtr(df, -abs(estimate) / error))
        p_greater = float(stdtr(df, -estimate / error))
    half_width = float(stdtrit(df, 1 - (1 - level) / 2)) * error
    return {
        "error": error,
        "df": float(df),
        "low": estimate - half_width,
        "high": estimate + half_width,
        "p_value": p_value,
        "p_greater": p_greater,
    }


def bootstrap_mean_test(
    values: np.ndarray, *, null: float, level: float, n_resamples: int, seed: int, block: int
) -> dict[str, float]:
    """Test that the mean of a series equals `null`, with a circular-block bootstrap standard error.

    The standard error of the mean comes from circular-block resamples (a bootstrap standard
    error with a t reference, not a bootstrap-t); the statistic
    ``(mean - null) / error`` is read against Student's t with ``n // block - 1`` degrees of
    freedom (the blocks less one), which keeps the size near its nominal level when the
    values overlap (e.g. 24-bar labels), where the normal reference rejects too often.

    Parameters
    ----------
    values : np.ndarray
        One value per period, in time order, shape ``(n,)``, at least two blocks.
    null : float
        The mean under the null hypothesis.
    level : float
        Coverage of the interval, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    block : int
        Block length in periods.

    Returns
    -------
    dict
        ``mean``, ``error``, ``df``, ``low`` and ``high`` (the t interval of the mean),
        ``p_value`` (two-sided) and ``p_greater`` (one-sided, H1: the mean is above `null`).

    Raises
    ------
    ValueError
        If the series has fewer than two blocks or `block` is out of range.
    """
    rng = np.random.default_rng(seed)
    error, df = _circular_error(values, rng, n_resamples=n_resamples, block=block)
    mean = float(np.mean(values))
    test = _t_test(mean - null, error, df=df, level=level)
    return {"mean": mean} | test | {"low": test["low"] + null, "high": test["high"] + null}


def _block_means(
    values: np.ndarray, rng: np.random.Generator, *, n_resamples: int, block: int
) -> np.ndarray:
    indices = block_indices(len(values), rng, n_resamples=n_resamples, block=block)
    return values[indices].mean(axis=1)


def bootstrap_interval(
    values: np.ndarray, *, level: float, n_resamples: int, seed: int, block: int
) -> tuple[float, float]:
    """Interval for the mean of `values` from a moving-block bootstrap (keeps autocorrelation).

    Parameters
    ----------
    values : np.ndarray
        One value per period, shape ``(n,)``.
    level : float
        Coverage, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    block : int
        Block length in periods (1 <= block <= n).

    Returns
    -------
    tuple of (float, float)
        Lower and upper bound.

    Raises
    ------
    ValueError
        If `block` is below 1 or longer than `values`.

    References
    ----------
    H. Künsch, "The Jackknife and the Bootstrap for General Stationary Observations", Annals of
    Statistics 17(3), 1989.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = _block_means(values, rng, n_resamples=n_resamples, block=block)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    level: float,
    n_resamples: int,
    seed: int,
    blocks: tuple[int, int],
) -> dict[str, float]:
    """Difference of the means of two separate periods, tested with bootstrap standard errors.

    Each period is resampled on its own with its own block length (the periods do not overlap
    in time and may be sampled at different spacings) to give the standard error of its mean;
    the difference is read against Student's t with Welch's degrees of freedom from the two
    periods' block counts.

    Parameters
    ----------
    first, second : np.ndarray
        One value per time of each period, in time order, shapes ``(n_first,)`` and
        ``(n_second,)``, each at least two blocks.
    level : float
        Coverage of the interval, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    blocks : tuple of (int, int)
        Block length in values of each period.

    Returns
    -------
    dict
        ``difference`` (mean of `first` minus mean of `second`), ``error``, ``df``, ``low``
        and ``high`` (the t interval), ``p_value`` (two-sided) and ``p_greater`` (one-sided,
        H1: the difference is above zero).

    Raises
    ------
    ValueError
        If a period has fewer than two blocks or a block is out of range.
    """
    rng = np.random.default_rng(seed)
    e1, d1 = _circular_error(first, rng, n_resamples=n_resamples, block=blocks[0])
    e2, d2 = _circular_error(second, rng, n_resamples=n_resamples, block=blocks[1])
    v1, v2 = e1**2, e2**2
    variance = v1 + v2
    # Welch's df; without spread (or when the squares underflow) the smaller block count
    df = min(d1, d2)
    welch = v1**2 / d1 + v2**2 / d2
    if variance > 0 and welch > 0:
        df = variance**2 / welch
    difference = float(np.mean(first) - np.mean(second))
    return {"difference": difference} | _t_test(difference, math.sqrt(variance), df=df, level=level)


def bootstrap_p_value(
    values: np.ndarray, *, null: float, n_resamples: int, seed: int, block: int
) -> float:
    """Two-sided p-value that the mean of `values` equals `null`, from a moving-block bootstrap.

    The draws are those of `bootstrap_interval` with the same `seed`; the p-value uses their
    standard error and the normal distribution, as `bootstrap_difference` does.

    Parameters
    ----------
    values : np.ndarray
        One value per period, in time order, shape ``(n,)``.
    null : float
        The mean under the null hypothesis, e.g. 0.5 for a hit rate.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    block : int
        Block length in periods (1 <= block <= n).

    Returns
    -------
    float
        The p-value, in [0, 1].

    Raises
    ------
    ValueError
        If `block` is below 1 or longer than `values`.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = _block_means(values, rng, n_resamples=n_resamples, block=block)
    return _normal_p_value(float(values.mean()) - null, float(means.std(ddof=1)))


def _normal_p_value(difference: float, error: float) -> float:
    # two-sided p-value of a zero difference from its bootstrap standard error; without spread
    # the difference is either exactly zero (p = 1) or certain (p = 0)
    if error > 0:
        return math.erfc(abs(difference / error) / math.sqrt(2))
    return 1.0 if difference == 0 else 0.0
