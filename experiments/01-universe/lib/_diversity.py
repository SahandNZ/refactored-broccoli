from statistics import NormalDist

import numpy as np
import pandas as pd

from src.stats import block_indices

from ._correlation import correlation_matrix
from ._settings import UniverseSettings

NEGATIVE_EIGENVALUE_TOLERANCE = -1e-8
"""Eigenvalues below this are reported as negative. Rounding in ``eigvalsh`` of a correlation
matrix of about 100 assets is near -1e-15, far above -1e-8; an eigenvalue of a pairwise
matrix that is truly negative is of the size of the correlation errors (about 0.01 and more)."""

BOOTSTRAP_RANDOM_UNIVERSES = 20
"""The Random universes ``Random 0`` to ``Random 19`` are the only ones that enter the median of
`gap_interval` and get a noise level (`noise_effective_rank`). Over all 1000 draws the bootstrap
would take about 20 minutes (1000 resamples times 1000 effective ranks) and the noise levels
about 80 seconds; with 20 both take under a minute. The point gap uses the median of all draws."""

NOISE_DRAWS = 10
"""Draws of `noise_effective_rank`; their mean varies by about 0.1 between seeds for 72 assets."""


def diversity(correlation: pd.DataFrame) -> dict[str, float]:
    """Measure how much independent behaviour a set of assets covers.

    The effective rank is that of O. Roy and M. Vetterli (2007), "The effective rank: a
    measure of effective dimensionality", EUSIPCO 2007. A correlation matrix built pair by pair
    (each pair on its own common weeks) can have small negative eigenvalues; the smallest is
    reported as ``min_eigenvalue`` and the negative ones count as 0 in the other measures.

    Parameters
    ----------
    correlation : pd.DataFrame
        Correlation matrix of the set, shape ``(n, n)``, without missing pairs.

    Returns
    -------
    dict
        ``mean_abs_correlation`` (off the diagonal), ``effective_rank`` (the exponential of the
        entropy of the normalised eigenvalues), ``components_80`` (principal components that
        explain 80 % of the variance), ``first_component_share`` and ``min_eigenvalue``.

    Raises
    ------
    ValueError
        If the matrix has a missing value.
    """
    matrix = correlation.to_numpy()
    if np.isnan(matrix).any():
        raise ValueError("the correlation matrix has missing pairs")
    size = len(matrix)
    off_diagonal = np.abs(matrix[~np.eye(size, dtype=bool)])
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    positive = np.clip(eigenvalues, 0.0, None)
    shares = positive / positive.sum()
    nonzero = shares[shares > 0]
    return {
        "mean_abs_correlation": float(off_diagonal.mean()) if size > 1 else 0.0,
        "effective_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
        "components_80": int(np.searchsorted(np.cumsum(shares), 0.8) + 1),
        "first_component_share": float(shares[0]),
        "min_eigenvalue": float(eigenvalues[-1]),
    }


def noise_effective_rank(returns: pd.DataFrame, *, min_overlap: int, seed: int) -> float:
    """Effective rank of independent Gaussian series with the missing data of `returns`.

    The noise level of a set: a window of few weeks caps the effective rank (``T`` weeks give a
    matrix of rank at most ``T``), so a level is read against this number of its own window.
    Each of `NOISE_DRAWS` draws puts standard normal values where `returns` has a return and
    NaN where it has none; the result is the mean effective rank of the draws.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns of the set, shape ``(n_weeks, n_assets)``, NaN where an asset has none.
    min_overlap : int
        Least number of common returns of a pair (`correlation_matrix`).
    seed : int
        Seed of the draws.

    Returns
    -------
    float
        The mean effective rank of the draws.
    """
    rng = np.random.default_rng(seed)
    has_return = returns.notna().to_numpy()
    ranks = np.empty(NOISE_DRAWS)
    for draw in range(NOISE_DRAWS):
        noise = np.where(has_return, rng.standard_normal(has_return.shape), np.nan)
        series = pd.DataFrame(noise, columns=returns.columns)
        ranks[draw] = diversity(correlation_matrix(series, min_overlap=min_overlap))[
            "effective_rank"
        ]
    return float(ranks.mean())


def gap_interval(
    returns: pd.DataFrame,
    sets: dict[str, list[str]],
    settings: UniverseSettings,
    *,
    gap: float,
    seed: int,
) -> tuple[float, float]:
    """Paired moving-block bootstrap interval of the universe's gap to the Random median.

    The gap is the universe's effective rank minus the median effective rank of the Random
    universes. Each of ``bootstrap_resamples`` resamples draws blocks of
    ``bootstrap_block_weeks`` consecutive weeks (``src.stats.block_indices``) and measures the
    universe and the first `BOOTSTRAP_RANDOM_UNIVERSES` Random universes on the same resampled
    weeks. The interval is the normal interval ``gap +- z * SE``: SE is the standard deviation
    of the resampled gaps and ``z`` the normal quantile of ``interval_level``. A percentile
    interval of the resampled gaps would be shifted: repeated weeks lower a high, tuned rank
    more than a Random one (the universe was picked on these weeks), so the point gap could lie
    above it. The selection is not redone inside a draw.

    Parameters
    ----------
    returns : pd.DataFrame
        Weekly returns, shape ``(n_weeks, n_assets)``, with a column for every member of the
        compared universes, NaN where an asset has none.
    sets : dict of str to list of str
        The assets of ``"Universe"`` and of each ``"Random <i>"``, in draw order (other
        universes are left out).
    settings : UniverseSettings
        ``bootstrap_min_overlap`` (least common returns of a pair in every resample),
        ``bootstrap_block_weeks``, ``bootstrap_resamples`` and ``interval_level`` (coverage,
        e.g. 0.95).
    gap : float
        The point gap, the centre of the interval.
    seed : int
        Seed of the draws.

    Returns
    -------
    tuple of (float, float)
        Lower and upper bound of the gap.

    Raises
    ------
    ValueError
        If a resample has a pair with fewer than ``bootstrap_min_overlap`` common returns, or
        the block is not between 1 and the number of weeks.
    """
    universe = sets["Universe"]
    randoms = [members for name, members in sets.items() if name.startswith("Random")]
    randoms = randoms[:BOOTSTRAP_RANDOM_UNIVERSES]
    columns = sorted(set(universe).union(*randoms))
    rng = np.random.default_rng(seed)
    positions = block_indices(
        len(returns), rng, n_resamples=settings.bootstrap_resamples,
        block=settings.bootstrap_block_weeks,
    )  # fmt: skip
    gaps = np.empty(settings.bootstrap_resamples)
    for draw, weeks in enumerate(positions):
        resampled = returns[columns].take(weeks)
        correlation = correlation_matrix(resampled, min_overlap=settings.bootstrap_min_overlap)
        rank = diversity(correlation.loc[universe, universe])["effective_rank"]
        random_ranks = [
            diversity(correlation.loc[random, random])["effective_rank"] for random in randoms
        ]
        gaps[draw] = rank - np.median(random_ranks)
    z = NormalDist().inv_cdf(1 - (1 - settings.interval_level) / 2)
    error = float(gaps.std(ddof=1))
    return gap - z * error, gap + z * error


def universe_beats_alternatives(table: pd.DataFrame, *, percentile: float) -> tuple[str, str]:
    """Answer 01-universe Q1 from the diversity rows of one window.

    Parameters
    ----------
    table : pd.DataFrame
        One row per universe of one window (column ``universe``: ``"Universe"``,
        ``"Liquid"``, ``"Random <i>"``) with ``size``, one ``size_<pool>`` column per pool,
        ``effective_rank``, ``noise_effective_rank`` and the universe's gap to the Random
        median with its bootstrap interval, ``gap``, ``gap_low`` and ``gap_high`` (NaN where
        none was measured: the gap is then left out).
    percentile : float
        The percentile of the random universes the universe must reach, e.g. 90.

    Returns
    -------
    tuple of (str, str)
        ``"yes"`` or ``"no"``, and the supporting numbers in one sentence.

    Raises
    ------
    ValueError
        If the table has no ``size_<pool>`` columns, or the Liquid or a Random universe has a
        different number of assets than the universe in some pool (the comparison would not
        be fair).
    """
    rows = table.set_index("universe")
    pools = [column for column in rows.columns if column.startswith("size_")]
    if not pools:
        raise ValueError("the table has no size_<pool> columns to compare the universes")
    sizes = rows[pools]
    different = [column for column in pools if sizes[column].nunique() != 1]
    if different:
        raise ValueError(f"the universes differ in size per pool: {sizes[different].to_dict()}")
    rank = rows["effective_rank"]
    random = rank[rank.index.str.startswith("Random")]
    threshold = float(np.percentile(random, percentile))
    passed = rank["Universe"] >= threshold and rank["Universe"] > rank["Liquid"]
    gap = rows["gap"].astype(float)["Universe"]
    low = rows["gap_low"].astype(float)["Universe"]
    high = rows["gap_high"].astype(float)["Universe"]
    interval = ""
    if not np.isnan(gap):
        interval = (
            f"; gap to the random median {gap:.2f} (paired block-bootstrap normal interval "
            f"{low:.2f} to {high:.2f}, in-sample)"
        )
    evidence = (
        f"effective rank of the universe {rank['Universe']:.2f} (noise level "
        f"{rows.loc['Universe', 'noise_effective_rank']:.2f}: independent series with its "
        f"missing data), Liquid {rank['Liquid']:.2f}, random {percentile:.0f}th "
        f"percentile {threshold:.2f} (median {random.median():.2f}, {len(random)} draws)"
        f"{interval}, {int(rows['size'].iloc[0])} assets each, the same number in every pool."
    )
    return ("yes" if passed else "no"), evidence
