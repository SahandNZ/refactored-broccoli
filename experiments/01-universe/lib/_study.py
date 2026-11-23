import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DataConfig, Instrument, read_manifest
from src.enums import Market

from ._candidates import (
    Measurement,
    data_filter,
    liquidity_filter,
    liquidity_table,
    measure_candidate,
    selection_pool,
)
from ._correlation import correlation_matrix, weekly_returns
from ._diversity import (
    BOOTSTRAP_RANDOM_UNIVERSES,
    NEGATIVE_EIGENVALUE_TOLERANCE,
    diversity,
    gap_interval,
    noise_effective_rank,
)
from ._selection import (
    SectorCap,
    drop_near_duplicates,
    greedy_order,
    liquid_pool,
    most_liquid,
    random_universe,
    take_with_sector_cap,
)
from ._settings import POOL_ORDER, UniverseSettings

logger = logging.getLogger(__name__)

SECTORS_FILE = Path(__file__).parent / "sectors.csv"
"""The GICS sector of each of the 128 stock candidates of ``configs/data.toml`` (MSCI and
S&P Global, Global Industry Classification Standard, 11 sectors), written in by hand from
the companies' published classifications; it caps the stocks per sector in the universe
(``max_per_sector``)."""


def select_universe(
    data: DataConfig, settings: UniverseSettings, *, seed: int
) -> dict[str, pd.DataFrame]:
    """Run the whole 01-universe selection on data before the development folds.

    Parameters
    ----------
    data : DataConfig
        ``configs/data.toml``: the lab's ``data/`` folder (``manifest.json`` lists the
        candidates), each candidate's instrument and the quality limits.
    settings : UniverseSettings
        The study's settings.
    seed : int
        Seed of the random comparison universes, the bootstrap interval and the noise level.

    Returns
    -------
    dict of str to pd.DataFrame
        ``universe`` (every candidate: market, source, last bar, the quality numbers of the
        selection window, first bar in it, ``passed_data_filter`` (it passed `data_filter` on
        the selection window), pool, pick ``order`` and ``in_universe``, dollar
        volume, spread, liquidity, why it was dropped or not picked, sector of a stock),
        ``correlation`` and ``holdout_correlation`` (the candidates after the filters,
        selection and held-out windows), ``diversity`` (one row per universe and window: the
        universe, Liquid and each Random draw, with the size of each pool, the noise level of
        the effective rank (`noise_effective_rank`) and, on the universe's row of the selection
        window, its gap to the Random median with the paired bootstrap interval
        (`gap_interval`); the noise level only for the universe, Liquid and the first
        `BOOTSTRAP_RANDOM_UNIVERSES` Random draws, NaN for the others), ``sector_mix``
        (`sector_mix`) and ``comparison_universes`` (the members of every compared universe:
        ``universe``, ``symbol``, ``pool``).

    Raises
    ------
    KeyError
        If a candidate of the manifest is not an instrument of `data`.
    ValueError
        If a kept candidate's pool has no quota, a pool mixes dollar volumes and spreads, a
        candidate has no liquidity measure, a stock has no sector, or a kept candidate's
        returns have zero variance or a pair too few common weekly returns in a window.
    """
    manifest = read_manifest(data.data_dir)
    instruments = _candidate_instruments(list(manifest.index), data)
    markets = pd.Series({symbol: instrument.market for symbol, instrument in instruments.items()})
    measured = {
        symbol: measure_candidate(data, instrument, settings)
        for symbol, instrument in instruments.items()
    }
    eligible, kept, table, dropped = _filter_candidates(markets, measured, settings)
    returns, correlations = _window_correlations(
        {symbol: measured[symbol].closes for symbol in kept}, settings
    )
    by_pool = _group_by_pool(kept, markets, settings.quotas, table)
    sectors = _stock_sectors(markets)
    cap = SectorCap(sectors, settings.max_per_sector)
    picks, not_picked = _pool_picks(correlations["selection"], by_pool, cap, settings)
    dropped.update(not_picked)
    sets = _comparison_sets(picks, by_pool, cap, settings.random_universes, seed=seed)
    universe = manifest[["market", "source", "last_bar"]].join(_quality_table(measured))
    universe = universe.join(_pick_table(by_pool, picks)).join(table)
    universe["passed_data_filter"] = universe.index.isin(eligible)
    universe["in_universe"] = universe.index.isin(sets["Universe"])
    universe["dropped"] = pd.Series(dropped)
    universe["sector"] = sectors
    table = _diversity_table(sets, correlations, by_pool)
    universe_row = (table["universe"] == "Universe") & (table["window"] == "selection")
    gap = float(table.loc[universe_row, "gap"].iloc[0])
    interval = gap_interval(returns["selection"], sets, settings, gap=gap, seed=seed)
    table.loc[universe_row, ["gap_low", "gap_high"]] = interval
    table["noise_effective_rank"] = _noise_levels(table, returns, settings, sets, seed=seed)
    return {
        "universe": universe.reset_index(names="symbol"),
        "correlation": correlations["selection"],
        "holdout_correlation": correlations["holdout"],
        "diversity": table,
        "sector_mix": sector_mix(universe),
        "comparison_universes": _members_table(sets, by_pool),
    }


def _candidate_instruments(symbols: list[str], data: DataConfig) -> dict[str, Instrument]:
    """Return the instrument of every manifest candidate.

    Raises
    ------
    KeyError
        If a candidate is not an instrument of `data`.
    """
    instruments = {instrument.name: instrument for instrument in data.instruments}
    unknown = sorted(set(symbols) - set(instruments))
    if unknown:
        raise KeyError(f"manifest candidates {unknown} are not instruments of configs/data.toml")
    return {symbol: instruments[symbol] for symbol in symbols}


def _window_correlations(
    closes: dict[str, pd.Series], settings: UniverseSettings
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return the weekly returns and the correlation matrix of the selection and held-out windows.

    Both dictionaries are keyed by window, ``"selection"`` and ``"holdout"``; each window has its
    own least overlap (``min_overlap``, ``holdout_min_overlap``). The run log names each window
    before its matrix is built, so an error of `correlation_matrix` is placed.
    """
    windows = {
        "selection": (settings.start, settings.end, settings.min_overlap),
        "holdout": (settings.end, settings.holdout_end, settings.holdout_min_overlap),
    }
    returns: dict[str, pd.DataFrame] = {}
    correlations: dict[str, pd.DataFrame] = {}
    for window, (start, end, min_overlap) in windows.items():
        logger.info("weekly correlations of the %s window [%s, %s)", window, start, end)
        returns[window] = weekly_returns(closes, start=start, end=end)
        correlations[window] = correlation_matrix(returns[window], min_overlap=min_overlap)
    return returns, correlations


def _filter_candidates(
    markets: pd.Series, measured: dict[str, Measurement], settings: UniverseSettings
) -> tuple[list[str], list[str], pd.DataFrame, dict[str, str]]:
    """Apply the data and liquidity filters.

    Returns the symbols that passed the data filter, the symbols kept after both filters, the
    liquidity table of the candidates that passed the data filter and the reason each dropped
    symbol was dropped.
    """
    eligible, dropped = data_filter(markets, measured, settings)
    table = liquidity_table({symbol: measured[symbol] for symbol in eligible})
    kept, thin = liquidity_filter(table, markets, settings)
    dropped.update(thin)
    logger.info(
        "%d candidates, %d pass the data filter, %d the liquidity filter",
        len(markets), len(eligible), len(kept),
    )  # fmt: skip
    return eligible, kept, table, dropped


def sector_mix(universe: pd.DataFrame) -> pd.DataFrame:
    """Count the stocks of the universe in each sector and stock pool.

    Parameters
    ----------
    universe : pd.DataFrame
        The ``universe`` table of `select_universe` (``market``, ``pool``, ``in_universe``,
        ``sector``).

    Returns
    -------
    pd.DataFrame
        One row per sector, one column per stock pool, counts of stocks in the universe.
    """
    stocks = universe[(universe["market"] == Market.STOCK.value) & universe["in_universe"]]
    return pd.crosstab(stocks["sector"], stocks["pool"])


def _stock_sectors(markets: pd.Series) -> pd.Series:
    """Return the sector of every stock of `markets` from `SECTORS_FILE`.

    Raises
    ------
    ValueError
        If a stock has no sector in the file.
    """
    sectors = pd.read_csv(SECTORS_FILE, index_col="symbol")["sector"]
    stocks = markets.index[markets == Market.STOCK]
    missing = sorted(set(stocks) - set(sectors.index))
    if missing:
        raise ValueError(f"stocks {missing} have no sector in {SECTORS_FILE.name}")
    return sectors[stocks]


def _quality_table(measured: dict[str, Measurement]) -> pd.DataFrame:
    """Return the quality numbers and the first bar of each candidate in the selection window."""
    rows = {
        symbol: {**measurement.quality, "first_bar": measurement.first_bar}
        for symbol, measurement in measured.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.drop(columns=["passed", "reason"], errors="ignore")


def _group_by_pool(
    symbols: list[str], markets: pd.Series, quotas: dict[str, int], table: pd.DataFrame
) -> dict[str, list[str]]:
    """Split `symbols` into the pools of `quotas` (`selection_pool`), each most liquid first.

    Every pool of `quotas` gets a list, empty when it has no symbol.

    Raises
    ------
    ValueError
        If a symbol's pool has no quota, or a pool mixes symbols ranked by dollar volume and
        by spread (the two scores cannot be compared).
    """
    by_pool: dict[str, list[str]] = {pool: [] for pool in quotas}
    for symbol in symbols:
        pool = selection_pool(symbol, markets[symbol])
        if pool not in by_pool:
            raise ValueError(f"pool {pool} of {symbol} has no quota in options.quotas")
        by_pool[pool].append(symbol)
    for pool, members in by_pool.items():
        if table.loc[members, "dollar_volume"].isna().nunique() > 1:
            raise ValueError(f"pool {pool} mixes dollar volumes and spreads")
        by_pool[pool] = most_liquid(table["liquidity"], members)
    return by_pool


def _pool_picks(
    correlation: pd.DataFrame,
    by_pool: dict[str, list[str]],
    cap: SectorCap,
    settings: UniverseSettings,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Pick the universe's assets of every pool and say why each other candidate was not picked.

    The pools are picked one after the other in `POOL_ORDER`. The greedy pick (`greedy_order`)
    runs inside the pool's liquid pool (`liquid_pool`), the pool's ``liquid_pool_factor`` times
    its quota, with at most `cap` stocks per sector; a candidate too close to an asset kept in
    any pool (the anchors of every pool, the picks of the pools done before, this pool's own
    picks) is a near-duplicate. A pool with fewer assets than its quota gives all it can; the
    log records the real count. A candidate that is not picked is a near-duplicate, outside the
    liquid pool, in a full sector, or not picked before the quota was filled.
    """
    picks: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    kept = [anchor for pool in by_pool for anchor in settings.anchors[pool]]
    for pool in sorted(by_pool, key=POOL_ORDER.index):
        symbols = by_pool[pool]
        quota, anchors = settings.quotas[pool], settings.anchors[pool]
        size = settings.liquid_pool_factor[pool] * quota
        candidates = liquid_pool(symbols, anchors=anchors, size=size)
        others = [name for name in candidates if name not in anchors]
        elsewhere = [symbol for symbol in kept if symbol not in symbols]
        others, duplicates = drop_near_duplicates(
            correlation, others, kept=elsewhere, threshold=settings.duplicate_threshold
        )
        picks[pool], own_duplicates = greedy_order(
            correlation, list(anchors) + others, anchors=anchors, size=quota,
            duplicate_threshold=settings.duplicate_threshold, cap=cap,
        )  # fmt: skip
        duplicates.update(own_duplicates)
        kept += [symbol for symbol in picks[pool] if symbol not in anchors]
        logger.info("%s: %d assets in the universe (quota %d)", pool, len(picks[pool]), quota)
        reasons.update(duplicates)
        for symbol in symbols:
            if symbol not in candidates:
                reasons[symbol] = "outside liquid pool"
            elif symbol not in picks[pool] and symbol not in duplicates:
                reasons[symbol] = _not_picked_reason(symbol, picks[pool], cap)
    return picks, reasons


def _not_picked_reason(symbol: str, pick: list[str], cap: SectorCap) -> str:
    """Say why a candidate of the liquid pool that is no near-duplicate was not picked."""
    if cap.is_full(symbol, pick):
        return f"not picked: sector cap ({cap.sectors[symbol]} full)"
    return "not picked"


def _pick_table(by_pool: dict[str, list[str]], picks: dict[str, list[str]]) -> pd.DataFrame:
    """Return the pool of every kept symbol and the pick order of the universe's assets."""
    rows = [
        {"symbol": symbol, "pool": pool, "order": pd.NA}
        for pool, symbols in by_pool.items()
        for symbol in symbols
    ]
    table = pd.DataFrame(rows, columns=["symbol", "pool", "order"]).set_index("symbol")
    for pick in picks.values():
        table.loc[pick, "order"] = range(1, len(pick) + 1)
    return table


def _comparison_sets(
    picks: dict[str, list[str]],
    by_pool: dict[str, list[str]],
    cap: SectorCap,
    random_universes: int,
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Build the universes Q1 compares, keyed by name.

    The universe; Liquid (the most liquid of each pool) and each Random draw, both with the
    universe's real count and the sector cap in each pool.
    """
    sizes = {pool: len(pick) for pool, pick in picks.items()}
    liquid = [
        symbol
        for pool, symbols in by_pool.items()
        for symbol in take_with_sector_cap(symbols, size=sizes[pool], cap=cap)
    ]
    rng = np.random.default_rng(seed)
    sets = {"Universe": [symbol for pick in picks.values() for symbol in pick], "Liquid": liquid}
    for draw in range(random_universes):
        sets[f"Random {draw}"] = random_universe(by_pool, sizes, cap=cap, rng=rng)
    return sets


def _diversity_table(
    sets: dict[str, list[str]],
    correlations: dict[str, pd.DataFrame],
    by_pool: dict[str, list[str]],
) -> pd.DataFrame:
    """Measure every universe in every window; warn when a matrix has a negative eigenvalue.

    Each row has the universe's size, its number of assets in each pool (``size_<pool>``) and
    the diversity measures; the universe's row of the selection window also has its ``gap``
    to the median effective rank of all Random universes (NaN in the other rows); the columns
    ``gap_low`` and ``gap_high`` of its interval stay NaN for the caller to fill. The warning
    counts only matrices with an eigenvalue below `NEGATIVE_EIGENVALUE_TOLERANCE`.
    """
    rows: list[dict[str, object]] = []
    for window, correlation in correlations.items():
        for name, members in sets.items():
            sizes = {
                f"size_{pool}": len(set(members) & set(symbols))
                for pool, symbols in by_pool.items()
            }
            measures = diversity(correlation.loc[members, members])
            rows.append({
                "universe": name, "window": window, "size": len(members), **sizes, **measures,
            })  # fmt: skip
    table = pd.DataFrame(rows)
    table[["gap", "gap_low", "gap_high"]] = float("nan")
    selection = table[table["window"] == "selection"].set_index("universe")["effective_rank"]
    random_median = selection[selection.index.str.startswith("Random")].median()
    universe_row = (table["universe"] == "Universe") & (table["window"] == "selection")
    table.loc[universe_row, "gap"] = selection["Universe"] - random_median
    negative = table[table["min_eigenvalue"] < NEGATIVE_EIGENVALUE_TOLERANCE]
    if len(negative):
        logger.warning(
            "%d correlation matrices have an eigenvalue below %g (smallest %.4f); "
            "negative eigenvalues count as 0",
            len(negative), NEGATIVE_EIGENVALUE_TOLERANCE, negative["min_eigenvalue"].min(),
        )  # fmt: skip
    return table


def _noise_levels(
    table: pd.DataFrame,
    returns: dict[str, pd.DataFrame],
    settings: UniverseSettings,
    sets: dict[str, list[str]],
    *,
    seed: int,
) -> list[float]:
    """Return the noise level of each row of the diversity `table` (`noise_effective_rank`).

    Only the universe, Liquid and the first `BOOTSTRAP_RANDOM_UNIVERSES` Random draws are
    measured (NaN for the other draws): a noise level takes about 40 ms per row and window.
    """
    measured = ["Universe", "Liquid"]
    measured += [f"Random {draw}" for draw in range(BOOTSTRAP_RANDOM_UNIVERSES)]
    min_overlaps = {"selection": settings.min_overlap, "holdout": settings.holdout_min_overlap}
    levels: list[float] = []
    for name, window in zip(table["universe"], table["window"], strict=True):
        if name in measured:
            members = returns[window][sets[name]]
            levels.append(
                noise_effective_rank(members, min_overlap=min_overlaps[window], seed=seed)
            )
        else:
            levels.append(float("nan"))
    return levels


def _members_table(sets: dict[str, list[str]], by_pool: dict[str, list[str]]) -> pd.DataFrame:
    """Return one row per member of every compared universe: ``universe``, ``symbol``, ``pool``."""
    pools = {symbol: pool for pool, symbols in by_pool.items() for symbol in symbols}
    rows = [
        {"universe": name, "symbol": symbol, "pool": pools[symbol]}
        for name, members in sets.items()
        for symbol in members
    ]
    return pd.DataFrame(rows, columns=["universe", "symbol", "pool"])
