"""The interval of a cell's coverage: a ratio of sums over decision times (rq2).

The coverage of a cell is a plain mean over its decisions, so its bootstrap resamples whole
decision times and divides the sums; `src.stats` has no ratio of sums, so this one helper
lives in the study's ``lib``. The centre test and the other intervals are
`src.stats.bootstrap_mean_test`.
"""

import numpy as np
from scipy.special import stdtrit

from src.stats import circular_block_indices


def ratio_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    level: float,
    n_resamples: int,
    seed: int,
    block: int,
) -> tuple[float, float]:
    """Return the t interval of ``sum(numerators) / sum(denominators)`` over decision times.

    Each draw takes circular blocks of decision times (`src.stats.circular_block_indices`)
    and divides the sums of the drawn times; the interval is the ratio plus or minus the
    standard deviation of the drawn ratios times Student's t quantile with
    ``n // block - 1`` degrees of freedom, as `src.stats.bootstrap_mean_test` gives the
    interval of a mean. With one decision per time it is that interval of the mean.

    Parameters
    ----------
    numerators, denominators : np.ndarray
        Per decision time in time order, e.g. the decisions with the label inside the
        interval and all decisions; shapes ``(n,)``, every denominator positive.
    level : float
        Coverage of the interval, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    seed : int
        Seed of the draws.
    block : int
        Block length in decision times, with at least two blocks (``2 * block <= n``).

    Returns
    -------
    tuple of (float, float)
        Lower and upper bound.

    Raises
    ------
    ValueError
        If the series has fewer than two blocks.
    """
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if len(numerators) < 2 * block:
        raise ValueError(f"need at least 2 blocks of {block} values, got {len(numerators)} values")
    rng = np.random.default_rng(seed)
    indices = circular_block_indices(len(numerators), rng, n_resamples=n_resamples, block=block)
    ratios = numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    ratio = float(numerators.sum() / denominators.sum())
    quantile = float(stdtrit(len(numerators) // block - 1, 1 - (1 - level) / 2))
    half_width = quantile * float(ratios.std(ddof=1))
    return ratio - half_width, ratio + half_width
