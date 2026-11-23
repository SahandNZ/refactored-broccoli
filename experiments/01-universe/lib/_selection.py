from dataclasses import dataclass

import numpy as np
import pandas as pd


def most_liquid(liquidity: pd.Series, candidates: list[str]) -> list[str]:
    """Return `candidates` from the most to the least liquid; ties keep the given order.

    Parameters
    ----------
    liquidity : pd.Series
        A liquidity score per symbol (higher is more liquid), covering `candidates`.
    candidates : list of str
        The symbols to sort.

    Returns
    -------
    list of str
        The same symbols, most liquid first.
    """
    return liquidity[candidates].sort_values(ascending=False, kind="stable").index.tolist()


@dataclass(frozen=True, slots=True)
class SectorCap:
    """At most `limit` picked stocks of one sector.

    Attributes
    ----------
    sectors : pd.Series
        The sector of each stock; a name without a sector (not a stock) is never capped.
    limit : int
        The most picked names of one sector.
    """

    sectors: pd.Series
    limit: int

    def is_full(self, candidate: str, picked: list[str]) -> bool:
        """Say whether `picked` already holds `limit` names of `candidate`'s sector."""
        sector = self.sectors.get(candidate)
        if not isinstance(sector, str):
            return False
        same = [name for name in picked if self.sectors.get(name) == sector]
        return len(same) >= self.limit


def take_with_sector_cap(ordered: list[str], *, size: int, cap: SectorCap) -> list[str]:
    """Take the first `size` names of `ordered`, skipping a name whose sector is full.

    Parameters
    ----------
    ordered : list of str
        Candidates in the order they should be taken.
    size : int
        How many to take (fewer when the list runs out).
    cap : SectorCap
        The sector cap.

    Returns
    -------
    list of str
        The names taken, in their order.
    """
    taken: list[str] = []
    for name in ordered:
        if len(taken) == size:
            break
        if not cap.is_full(name, taken):
            taken.append(name)
    return taken


def liquid_pool(candidates: list[str], *, anchors: tuple[str, ...], size: int) -> list[str]:
    """Return the liquid pool of one selection pool: the anchors, then the most liquid others.

    Parameters
    ----------
    candidates : list of str
        The pool's candidates that passed the filters, most liquid first (`most_liquid`).
    anchors : tuple of str
        Assets kept for their economic role (must be among `candidates`).
    size : int
        How many candidates the liquid pool holds, anchors included (all when there are fewer).

    Returns
    -------
    list of str
        The anchors first, then the most liquid other candidates.

    Raises
    ------
    ValueError
        If an anchor is not a candidate.
    """
    missing = [anchor for anchor in anchors if anchor not in candidates]
    if missing:
        raise ValueError(f"anchors {missing} are not among the candidates")
    others = [name for name in candidates if name not in anchors]
    return (list(anchors) + others)[: max(size, len(anchors))]


def greedy_order(
    correlation: pd.DataFrame,
    pool: list[str],
    *,
    anchors: tuple[str, ...],
    size: int,
    duplicate_threshold: float,
    cap: SectorCap,
) -> tuple[list[str], dict[str, str]]:
    """Pick up to `size` assets of one liquid pool by the low-correlation rule.

    The anchors come first, in their order (with no anchors, the first of `pool`, the most
    liquid, starts). Then, repeatedly, the candidate with the lowest mean absolute correlation
    to the assets already kept is added; ties go to the more liquid one. A candidate whose
    absolute correlation to a kept asset is above `duplicate_threshold` is dropped as a
    near-duplicate of it (`drop_near_duplicates` checks the assets kept in other pools first).
    A candidate whose sector is full (`cap`) is skipped, not dropped.
    The pick stops at `size` assets or when no candidate is left.

    Parameters
    ----------
    correlation : pd.DataFrame
        Correlation matrix covering at least `pool`, without missing pairs.
    pool : list of str
        The liquid pool (`liquid_pool`): the anchors, then the others most liquid first.
    anchors : tuple of str
        Assets kept first for their economic role (in `pool`).
    size : int
        How many assets to pick, anchors included.
    duplicate_threshold : float
        Absolute correlation above which a candidate is a near-duplicate.
    cap : SectorCap
        The most picked stocks per sector.

    Returns
    -------
    tuple of (list of str, dict of str to str)
        The pick in its order, and the reason each dropped candidate was dropped.

    Raises
    ------
    ValueError
        If `size` is below the number of anchors or a pair of the pool has no correlation.
    """
    if size < len(anchors):
        raise ValueError(f"size must be >= {len(anchors)} anchors, got {size}")
    if not pool or size == 0:
        return [], {}
    absolute = correlation.loc[pool, pool].abs()
    if absolute.isna().to_numpy().any():
        raise ValueError("the correlation matrix has missing pairs in the pool")
    kept = list(anchors) or pool[:1]
    remaining = [name for name in pool if name not in kept]
    dropped: dict[str, str] = {}
    for asset in kept:
        remaining, duplicates = _drop_duplicates(absolute[asset], remaining, duplicate_threshold)
        dropped.update(duplicates)
    while remaining and len(kept) < size:
        allowed = [name for name in remaining if not cap.is_full(name, kept)]
        if not allowed:
            break
        best = str(absolute.loc[allowed, kept].mean(axis=1).idxmin())
        kept.append(best)
        remaining.remove(best)
        remaining, duplicates = _drop_duplicates(absolute[best], remaining, duplicate_threshold)
        dropped.update(duplicates)
    return kept, dropped


def drop_near_duplicates(
    correlation: pd.DataFrame, candidates: list[str], *, kept: list[str], threshold: float
) -> tuple[list[str], dict[str, str]]:
    """Drop every candidate whose absolute correlation to an asset of `kept` is above `threshold`.

    The near-duplicate rule across pools: `kept` holds the assets already kept in other pools.

    Parameters
    ----------
    correlation : pd.DataFrame
        Correlation matrix covering `candidates` and `kept`, without missing pairs.
    candidates : list of str
        The names to check, in their order.
    kept : list of str
        Assets already kept.
    threshold : float
        Absolute correlation above which a candidate is a near-duplicate.

    Returns
    -------
    tuple of (list of str, dict of str to str)
        The candidates left, in their order, and the reason each dropped one was dropped.

    Raises
    ------
    ValueError
        If a pair of a candidate and a kept asset has no correlation.
    """
    absolute = correlation.loc[candidates, kept].abs()
    if absolute.isna().to_numpy().any():
        raise ValueError("the correlation matrix has missing pairs between pools")
    remaining = list(candidates)
    dropped: dict[str, str] = {}
    for asset in kept:
        remaining, duplicates = _drop_duplicates(absolute[asset], remaining, threshold)
        dropped.update(duplicates)
    return remaining, dropped


def _drop_duplicates(
    to_kept: pd.Series, remaining: list[str], threshold: float
) -> tuple[list[str], dict[str, str]]:
    """Split off the near-duplicates of one kept asset from `remaining`, with their reasons.

    `to_kept` is the absolute correlation of every candidate to that kept asset.
    """
    values = to_kept[remaining]
    duplicates = {
        str(name): f"near-duplicate of {to_kept.name} ({value:.2f})"
        for name, value in values[values > threshold].items()
    }
    return [name for name in remaining if name not in duplicates], duplicates


def random_universe(
    candidates: dict[str, list[str]],
    sizes: dict[str, int],
    *,
    cap: SectorCap,
    rng: np.random.Generator,
) -> list[str]:
    """Draw a random universe with the given size per pool and the sector cap.

    Parameters
    ----------
    candidates : dict of str to list of str
        Candidates per pool (after the data and liquidity filters).
    sizes : dict of str to int
        Assets per pool: the size of the rule-based universe's pick in that pool.
    cap : SectorCap
        The most drawn stocks per sector in one pool.
    rng : np.random.Generator
        The random source (seeded by the caller).

    Returns
    -------
    list of str
        The drawn assets.
    """
    chosen: list[str] = []
    for pool, names in candidates.items():
        shuffled = [str(name) for name in rng.permutation(names)]
        chosen.extend(take_with_sector_cap(shuffled, size=sizes[pool], cap=cap))
    return chosen
