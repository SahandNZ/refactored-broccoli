"""The block bootstrap of one book: its mean net return and its break-even cost (rq3).

The scheme of rq1, rq2 and rq4: circular blocks (`src.stats.circular_block_indices`) and the
mean net return tested as `src.stats.bootstrap_mean_test` tests a mean (the bootstrap
standard error against Student's t with the blocks less one as degrees of freedom, and the
matching t interval). One block length for the book's net and gross returns and turnover
(`book_block`: the longest of their `src.stats.politis_white_block` lengths over the
effective times, with rq1's floor), and the same draws give the percentile interval of the
break-even cost (`book_bootstrap`).
"""

import math

import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit

from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    circular_block_indices,
    has_enough_blocks,
    politis_white_block,
)

BOOTSTRAP_COLUMNS = (
    "net_low",
    "net_high",
    "p_value",
    "break_even_low",
    "break_even_high",
)
"""The keys of `book_bootstrap`."""

BOOK_SERIES = ("net", "gross", "turnover")
"""The series of a book that `book_bootstrap` resamples."""


def is_effective(book: pd.DataFrame, *, horizon: int) -> pd.Series:
    """Return which times of a book count in its effective sample.

    At horizon 1 every book time counts (a flat book is a real zero return). A book of
    horizon ``h > 1`` holds each position for ``h`` times with one return, so the times in
    between are exactly zero in every series (net too: no return and no trade) and do not
    count: only the times with a return or a trade do. A book time whose gross
    return is exactly 0, without a trade, does not count either (e.g. a roll whose returns
    sum to 0).

    Parameters
    ----------
    book : pd.DataFrame
        The book: ``gross`` and ``turnover`` per book time, shape ``(n_times, ...)``.
    horizon : int
        The book's horizon in own bars.

    Returns
    -------
    pd.Series
        Booleans, shape ``(n_times,)``.
    """
    if horizon == 1:
        return pd.Series(True, index=book.index)
    return (book["gross"] != 0) | (book["turnover"] != 0)


def _effective_series(book: pd.DataFrame, *, horizon: int) -> list[np.ndarray]:
    # the net and gross returns and turnover at the book's effective times
    effective = book[is_effective(book, horizon=horizon)]
    return [effective[name].to_numpy(dtype=float) for name in BOOK_SERIES]


def book_block(book: pd.DataFrame, *, horizon: int, overlap: int) -> int:
    """Return the block length of a book's bootstrap, in effective times.

    The longest `src.stats.politis_white_block` (Politis and White 2004 with the correction
    of Patton, Politis and White 2009) of the book's net and gross returns and turnover at
    its effective times (`is_effective`), so the break-even interval keeps the
    autocorrelation of the most persistent of them, with the floor of rq1: at least
    `src.stats.BLOCK_OVERLAP_FLOOR` (4) times `overlap` and at least 2.

    Parameters
    ----------
    book : pd.DataFrame
        The book in time order: ``net``, ``gross`` and ``turnover`` per book time.
    horizon : int
        The book's horizon in own bars.
    overlap : int
        How many later effective times a position spans (the median over the book's
        positions): 0 when every asset's positions open and close at the same times, more
        when they do not (a market that pools sessions, or chains out of phase).

    Returns
    -------
    int
        The block length, at least 2.
    """
    lengths = [politis_white_block(series) for series in _effective_series(book, horizon=horizon)]
    return max(*lengths, BLOCK_OVERLAP_FLOOR * overlap, 2)


def book_bootstrap(
    book: pd.DataFrame,
    *,
    horizon: int,
    block: int,
    level: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap the mean net return and the break-even cost of one book.

    The bootstrap resamples the book's effective times (`is_effective`) in circular blocks of
    `block` (`book_block`, `src.stats.circular_block_indices`): the zero times of a book of
    horizon ``h > 1`` would hide the autocorrelation of its returns. One set of draws gives
    both intervals. The mean net return is tested as `src.stats.bootstrap_mean_test` tests a
    mean, the method of rq1, rq2 and rq4: its bootstrap standard error against Student's t
    with ``n // block - 1`` degrees of freedom (``n`` effective times), the one-sided p-value
    of H1 "above 0" and the t interval at `level`; the interval is scaled from the effective
    times to all the book's times (the left-out times are 0), so it is per book time, as the
    book's own mean, and the p-value does not depend on the zero times. The break-even cost
    of each draw is its mean gross return over its mean turnover; its interval is the
    percentile interval at `level` (the owner's rule for rq3: a ratio skewed at low
    turnover), NaN when a draw has no turnover (its break-even is undefined).

    Parameters
    ----------
    book : pd.DataFrame
        The book in time order: ``net``, ``gross`` and ``turnover`` per book time, shape
        ``(n_times, ...)``.
    horizon : int
        The book's horizon in own bars (`is_effective`).
    block : int
        The block length in effective times (`book_block`); the effective times must hold
        at least `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks (`src.stats.has_enough_blocks`).
    level : float
        Coverage of the intervals, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    rng : np.random.Generator
        The cell's random stream.

    Returns
    -------
    dict of str to float
        ``net_low`` and ``net_high`` (per book time), ``p_value`` (one-sided),
        ``break_even_low`` and ``break_even_high`` (bps per side).

    Raises
    ------
    ValueError
        If the effective times hold fewer than `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks.
    """
    net, gross, turnover = _effective_series(book, horizon=horizon)
    if not has_enough_blocks(len(net), block):
        raise ValueError(
            f"{len(net)} effective times hold fewer than {MIN_BOOTSTRAP_BLOCKS} blocks of {block}"
        )
    share = len(net) / len(book)
    indices = circular_block_indices(len(net), rng, n_resamples=n_resamples, block=block)
    error = float(net[indices].mean(axis=1).std(ddof=1))
    df = len(net) // block - 1
    tail = (1 - level) / 2
    mean = float(net.mean())
    half_width = float(stdtrit(df, 1 - tail)) * error
    p_value = float(stdtr(df, -mean / error)) if error > 0 else (0.0 if mean > 0 else 1.0)
    turnover_means = turnover[indices].mean(axis=1)
    break_even = [math.nan, math.nan]
    if (turnover_means > 0).all():
        ratios = gross[indices].mean(axis=1) / turnover_means * 1e4
        break_even = [float(np.quantile(ratios, tail)), float(np.quantile(ratios, 1 - tail))]
    return {
        "net_low": (mean - half_width) * share,
        "net_high": (mean + half_width) * share,
        "p_value": p_value,
        "break_even_low": break_even[0],
        "break_even_high": break_even[1],
    }
