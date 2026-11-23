from dataclasses import dataclass, fields
from typing import Any

POOL_ORDER = ("crypto", "forex", "commodity", "index", "stock_us", "stock_non_us")
"""Every selection pool, in the order the universe is picked (`selection_pool` names them)."""


@dataclass(frozen=True, slots=True)
class UniverseSettings:
    """The ``[options]`` of 01-universe's ``study.toml``, checked and typed.

    Attributes
    ----------
    start, end : str
        The selection window ``[start, end)``: every filter, liquidity score and the
        correlations of the pick use only these dates.
    holdout_end : str
        End of the held-out window ``[end, holdout_end)`` on which the diversity of the same
        universes is measured again; Q1 is decided on this window.
    start_tolerance_days : int
        How many days after ``start`` a first bar still counts as "data from the start".
    listed_by : dict of str to str
        Latest first bar per market that may start later than ``start`` (no tolerance added).
    pegged_currencies : tuple of str
        Forex pairs with one of these currencies are dropped.
    min_daily_amount : float
        Least median daily quote volume of a crypto pair.
    max_spread_ratio : float
        Largest median spread, as a multiple of the median of the asset's market.
    min_overlap : int
        Least number of common weekly returns of a pair for its correlation in the selection
        window.
    holdout_min_overlap : int
        The same for the shorter held-out window.
    bootstrap_min_overlap : int
        The same for each bootstrap draw of the selection window (a draw holds fewer weeks of
        a late-starting asset than the window, by chance).
    duplicate_threshold : float
        Absolute correlation above which a candidate is a near-duplicate of an asset kept in
        any pool.
    liquid_pool_factor : dict of str to int
        Per pool: the greedy pick of the pool chooses among this many times its quota of
        liquid assets.
    max_per_sector : int
        The most stocks of one GICS sector in each stock pool.
    anchors : dict of str to tuple of str
        Assets each pool starts from.
    quotas : dict of str to int
        The most assets of each pool in the universe.
    random_universes : int
        How many random comparison universes are drawn.
    q1_percentile : float
        The percentile of the random universes the universe must reach in Q1.
    bootstrap_block_weeks : int
        Block length, in weeks, of the bootstrap interval of the universe's gap to the median
        effective rank of the Random universes.
    bootstrap_resamples : int
        Bootstrap draws of that interval.
    interval_level : float
        Coverage of that interval, e.g. 0.95.
    """

    start: str
    end: str
    holdout_end: str
    start_tolerance_days: int
    listed_by: dict[str, str]
    pegged_currencies: tuple[str, ...]
    min_daily_amount: float
    max_spread_ratio: float
    min_overlap: int
    holdout_min_overlap: int
    bootstrap_min_overlap: int
    duplicate_threshold: float
    liquid_pool_factor: dict[str, int]
    max_per_sector: int
    anchors: dict[str, tuple[str, ...]]
    quotas: dict[str, int]
    random_universes: int
    q1_percentile: float
    bootstrap_block_weeks: int
    bootstrap_resamples: int
    interval_level: float


def read_settings(options: dict[str, object]) -> UniverseSettings:
    """Read the study's ``[options]`` into a `UniverseSettings`.

    Parameters
    ----------
    options : dict
        The ``[options]`` table of ``study.toml`` (`src.experiment.StudyConfig.options`); its
        ``ranking`` table (the settings of Q3) is left out.

    Returns
    -------
    UniverseSettings
        The typed settings.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults) or unknown, the pools of the anchors,
        the quotas and the liquid pool factors differ, or a pool is not one of `POOL_ORDER`.
    """
    # the ranking table holds the settings of Q3 (`read_ranking_settings`), not of the selection
    selection = {name: value for name, value in options.items() if name != "ranking"}
    expected = {field.name for field in fields(UniverseSettings)}
    missing, unknown = sorted(expected - set(selection)), sorted(set(selection) - expected)
    if missing or unknown:
        raise KeyError(f"options missing {missing}, unknown {unknown}")
    # the [options] table is untyped TOML; every value is converted to its type here
    raw: dict[str, Any] = selection
    if set(raw["anchors"]) != set(raw["quotas"]):
        raise KeyError(f"anchors {sorted(raw['anchors'])} and quotas {sorted(raw['quotas'])}")
    if set(raw["liquid_pool_factor"]) != set(raw["quotas"]):
        raise KeyError(
            f"liquid_pool_factor {sorted(raw['liquid_pool_factor'])} and quotas "
            f"{sorted(raw['quotas'])}"
        )
    unknown_pools = sorted(set(raw["quotas"]) - set(POOL_ORDER))
    if unknown_pools:
        raise KeyError(f"pools {unknown_pools} are not among {POOL_ORDER}")
    return UniverseSettings(
        start=str(raw["start"]),
        end=str(raw["end"]),
        holdout_end=str(raw["holdout_end"]),
        start_tolerance_days=int(raw["start_tolerance_days"]),
        listed_by={str(market): str(date) for market, date in raw["listed_by"].items()},
        pegged_currencies=tuple(str(code) for code in raw["pegged_currencies"]),
        min_daily_amount=float(raw["min_daily_amount"]),
        max_spread_ratio=float(raw["max_spread_ratio"]),
        min_overlap=int(raw["min_overlap"]),
        holdout_min_overlap=int(raw["holdout_min_overlap"]),
        bootstrap_min_overlap=int(raw["bootstrap_min_overlap"]),
        duplicate_threshold=float(raw["duplicate_threshold"]),
        liquid_pool_factor={
            str(pool): int(factor) for pool, factor in raw["liquid_pool_factor"].items()
        },
        max_per_sector=int(raw["max_per_sector"]),
        anchors={
            str(pool): tuple(str(name) for name in anchors)
            for pool, anchors in raw["anchors"].items()
        },
        quotas={str(pool): int(quota) for pool, quota in raw["quotas"].items()},
        random_universes=int(raw["random_universes"]),
        q1_percentile=float(raw["q1_percentile"]),
        bootstrap_block_weeks=int(raw["bootstrap_block_weeks"]),
        bootstrap_resamples=int(raw["bootstrap_resamples"]),
        interval_level=float(raw["interval_level"]),
    )
