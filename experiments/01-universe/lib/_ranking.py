"""Q3 of 01-universe: the stability of the ranking of the models by mean CRPS.

The models are scored on the forecast cache of 02-forecasts, on one horizon of its test
windows: every model on the same decisions, those every compared model scored. A part of the
universe (a random half of each pool, one market, one test quarter) is scored by the mean over
its assets of each asset's mean CRPS, so every asset weighs the same. Each part is compared with
the rest of the universe (the other half, the other markets, the other quarters) by Kendall's
tau of the two rankings, so the two rankings never share a decision.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark import ForecastCache, decision_losses, scored_decisions

PART_KEYS = ["time", "asset", "window"]
"""The columns that name one scored decision of the comparison (one horizon)."""


@dataclass(frozen=True, slots=True)
class RankingSettings:
    """The ``[options.ranking]`` of 01-universe's ``study.toml``: the settings of Q3.

    Attributes
    ----------
    forecast_cache : str
        The run id of the 02-forecasts cache whose forecasts are ranked.
    windows : tuple of str
        The scored windows of the cache (the test quarters t1-t4).
    horizon : int
        The scored horizon, in the asset's own bars.
    splits : int
        How many random splits of the universe into two halves are drawn.
    tau_threshold : float
        Q3 is yes when the median Kendall tau of the splits is at least this.
    range_level : float
        The central share of the split taus that their reported range covers (not an
        interval).
    """

    forecast_cache: str
    windows: tuple[str, ...]
    horizon: int
    splits: int
    tau_threshold: float
    range_level: float


def read_ranking_settings(options: dict[str, Any]) -> RankingSettings:
    """Read ``options["ranking"]`` of the study into a `RankingSettings`.

    Parameters
    ----------
    options : dict
        The ``[options]`` table of ``study.toml``.

    Returns
    -------
    RankingSettings
        The typed settings.

    Raises
    ------
    KeyError
        If the ranking table or one of its keys is missing, or a key is unknown.
    ValueError
        If there are fewer than two windows or no split.
    """
    raw: dict[str, Any] = options["ranking"]
    expected = set(RankingSettings.__dataclass_fields__)
    missing, unknown = sorted(expected - set(raw)), sorted(set(raw) - expected)
    if missing or unknown:
        raise KeyError(f"options.ranking missing {missing}, unknown {unknown}")
    settings = RankingSettings(
        forecast_cache=str(raw["forecast_cache"]),
        windows=tuple(str(window) for window in raw["windows"]),
        horizon=int(raw["horizon"]),
        splits=int(raw["splits"]),
        tau_threshold=float(raw["tau_threshold"]),
        range_level=float(raw["range_level"]),
    )
    if len(settings.windows) < 2 or settings.splits < 1 or not 0 < settings.range_level < 1:
        raise ValueError(
            f"Q3 needs two windows or more, one split or more and a range level in (0, 1), got "
            f"{settings.windows}, {settings.splits}, {settings.range_level}"
        )
    return settings


def common_losses(
    cache: ForecastCache, models: list[str], *, windows: list[str], horizon: int
) -> pd.DataFrame:
    """Return the CRPS of every model on the decisions that every one of `models` scored.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The compared models.
    windows : list of str
        The scored windows.
    horizon : int
        The scored horizon.

    Returns
    -------
    pd.DataFrame
        ``PART_KEYS``, ``market`` and one CRPS column per model, one row per decision.

    Raises
    ------
    ValueError
        If no decision is scored by every model.
    """
    common: pd.DataFrame | None = None
    for model in models:
        decisions, _ = scored_decisions(cache, model, windows=windows)
        decisions = decisions[decisions["horizon"] == horizon]
        losses = decision_losses(decisions)[[*PART_KEYS, "market", "crps"]]
        losses = losses.rename(columns={"crps": model})
        if common is None:
            common = losses
        else:
            common = common.merge(losses, on=[*PART_KEYS, "market"], validate="one_to_one")
    if common is None or common.empty:
        raise ValueError(f"no decision of horizon {horizon} in {windows} is scored by {models}")
    return common.sort_values(PART_KEYS, ignore_index=True)


def asset_means(losses: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Return each asset's mean CRPS per model over the decisions of `losses`.

    Parameters
    ----------
    losses : pd.DataFrame
        `common_losses` or a part of it.
    models : list of str
        The model columns.

    Returns
    -------
    pd.DataFrame
        One row per asset (index ``asset``), one column per model.
    """
    return losses.groupby("asset")[models].mean()


def kendall(first: pd.Series, second: pd.Series) -> float:
    """Return Kendall's tau-b between two score vectors of the same models.

    Parameters
    ----------
    first, second : pd.Series
        Scores indexed by model (lower is better; the rank order is what counts).

    Returns
    -------
    float
        Tau in [-1, 1].

    Raises
    ------
    ValueError
        If the two vectors do not hold the same models or hold fewer than two.
    """
    if set(first.index) != set(second.index) or len(first) < 2:
        raise ValueError(f"tau needs the same two models or more, got {list(first.index)}")
    second = second.reindex(first.index)
    # pandas' Kendall correlation is tau-b (ties counted as scipy.stats.kendalltau does)
    return float(first.corr(second, method="kendall"))


def agreement(first: pd.Series, second: pd.Series) -> dict[str, object]:
    """Return how far two score vectors of the same models agree on their ranking.

    Parameters
    ----------
    first, second : pd.Series
        Scores indexed by model (lower is better).

    Returns
    -------
    dict
        ``tau`` (`kendall`), ``same_top`` (the same model ranks first in both) and
        ``same_top3`` (the same three models rank first, in the same order).
    """
    first_order = first.sort_values().index[:3].tolist()
    second_order = second.reindex(first.index).sort_values().index[:3].tolist()
    return {
        "tau": kendall(first, second),
        "same_top": first_order[0] == second_order[0],
        "same_top3": first_order == second_order,
    }


def random_half(pools: pd.Series, rng: np.random.Generator) -> pd.Index:
    """Return a random half of the assets: half of every pool's assets, rounded down.

    Parameters
    ----------
    pools : pd.Series
        The selection pool of every asset, indexed by asset.
    rng : np.random.Generator
        The random stream of the draw.

    Returns
    -------
    pd.Index
        The assets of the half; the other assets are the other half.

    Raises
    ------
    ValueError
        If an asset has no pool or a pool has fewer than two assets.
    """
    if pools.isna().any():
        raise ValueError(f"assets without a pool: {sorted(pools[pools.isna()].index)}")
    members = [group.index.to_numpy() for _, group in pools.groupby(pools, sort=True)]
    small = sorted(str(pools[assets[0]]) for assets in members if len(assets) < 2)
    if small:
        raise ValueError(f"pools {small} have fewer than two assets and cannot be split")
    half = [rng.permutation(assets)[: len(assets) // 2] for assets in members]
    return pd.Index(np.concatenate(half), name=pools.index.name)


def split_draws(
    means: pd.DataFrame, pools: pd.Series, *, splits: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return the `agreement` of the two halves of each random split of the universe.

    Each split puts a `random_half` of the assets (half of every pool) into the first half and
    the rest into the second, so both halves keep the pools' shares; each half is scored by the
    mean of its assets' mean CRPS.

    Parameters
    ----------
    means : pd.DataFrame
        `asset_means` over the scored windows, one row per asset.
    pools : pd.Series
        The selection pool of every asset of `means`, indexed by asset.
    splits : int
        How many splits are drawn.
    rng : np.random.Generator
        The random stream of the draws.

    Returns
    -------
    pd.DataFrame
        ``split`` and the columns of `agreement`, one row per split.

    Raises
    ------
    ValueError
        If an asset has no pool or a pool has fewer than two assets.
    """
    pools = pools.reindex(means.index)
    rows: list[dict[str, object]] = []
    for draw in range(splits):
        is_first = means.index.isin(random_half(pools, rng))
        rows.append({"split": draw} | agreement(means[is_first].mean(), means[~is_first].mean()))
    return pd.DataFrame(rows, columns=["split", "tau", "same_top", "same_top3"])


def split_summary(splits: pd.DataFrame, group: str, *, level: float) -> tuple[float, float, float]:
    """Return the median tau of one group's splits and the central `level` range of the taus.

    Parameters
    ----------
    splits : pd.DataFrame
        ``ranking_splits`` of `ranking_tables` (``group``, ``tau``).
    group : str
        ``"hourly"`` or ``"grid"``.
    level : float
        The central share of the taus the range covers, e.g. 0.95.

    Returns
    -------
    tuple of float
        The median, the low and the high end of the range.

    Raises
    ------
    ValueError
        If the group has no split.
    """
    taus = splits.loc[splits["group"] == group, "tau"].to_numpy(dtype=float)
    if len(taus) == 0:
        raise ValueError(f"no split of group {group}")
    low, high = np.quantile(taus, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(taus)), float(low), float(high)


def part_taus(losses: pd.DataFrame, models: list[str], *, column: str) -> pd.DataFrame:
    """Return the tau of each value of `column` (a market, a window) against all the others.

    Parameters
    ----------
    losses : pd.DataFrame
        `common_losses`.
    models : list of str
        The model columns.
    column : str
        ``"market"`` or ``"window"``.

    Returns
    -------
    pd.DataFrame
        ``part`` (the value), ``assets`` (its number of assets), ``decisions`` and the
        columns of `agreement`, one row per value.
    """
    rows: list[dict[str, object]] = []
    for part in sorted(str(value) for value in losses[column].unique()):
        is_part = (losses[column] == part).to_numpy()
        inside = asset_means(pd.DataFrame(losses[is_part]), models).mean()
        outside = asset_means(pd.DataFrame(losses[~is_part]), models).mean()
        assets = int(losses.loc[is_part, "asset"].nunique())
        row: dict[str, object] = {"part": part, "assets": assets, "decisions": int(is_part.sum())}
        rows.append(row | agreement(inside, outside))
    return pd.DataFrame(
        rows, columns=["part", "assets", "decisions", "tau", "same_top", "same_top3"]
    )


def model_scores(losses: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Return every model's score on the whole universe and its rank (1 = lowest CRPS).

    Parameters
    ----------
    losses : pd.DataFrame
        `common_losses`.
    models : list of str
        The model columns.

    Returns
    -------
    pd.DataFrame
        ``model``, ``crps`` (the mean over assets of each asset's mean CRPS) and ``rank``.
    """
    scores = asset_means(losses, models).mean()
    ranks = scores.rank(method="min").astype(int)
    return pd.DataFrame({"model": scores.index, "crps": scores.to_numpy(), "rank": ranks})


def _group_tables(
    losses: pd.DataFrame, models: list[str], pools: pd.Series, *, group: str, splits: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # the scores, the part taus and the split taus of one group of compared models
    scores = model_scores(losses, models).assign(group=group)
    rng = np.random.default_rng(seed)
    split_rows = split_draws(asset_means(losses, models), pools, splits=splits, rng=rng)
    split_rows = split_rows.assign(group=group)
    parts = pd.concat(
        [
            part_taus(losses, models, column="market").assign(check="market"),
            part_taus(losses, models, column="window").assign(check="quarter"),
        ],
        ignore_index=True,
    ).assign(group=group)
    scores = scores.assign(decisions=len(losses), assets=int(losses["asset"].nunique()))
    return scores, parts, split_rows


def ranking_tables(
    cache: ForecastCache,
    models: list[str],
    pools: pd.Series,
    settings: RankingSettings,
    *,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Return the tables of Q3 for the hourly models and for every model on the grid points.

    The hourly group holds the models of `models` that forecast every decision (``every_hours``
    0 in the cache); the grid group holds every model of `models`, on the decisions the grid
    models (Kronos) made. Each group's split draws start from the same `seed`.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The compared models (study.toml ``[models]``).
    pools : pd.Series
        The selection pool of every asset of the cache, indexed by asset.
    settings : RankingSettings
        The windows, horizon and number of splits.
    seed : int
        The seed of the split draws.

    Returns
    -------
    dict of str to pd.DataFrame
        ``ranking_scores`` (``group``, ``model``, ``crps``, ``rank``, ``decisions``,
        ``assets``, ``kind``), ``ranking_parts`` (``group``, ``check`` and the columns of
        `part_taus`) and ``ranking_splits`` (``group`` and the columns of `split_draws`).

    Raises
    ------
    ValueError
        If the cache's assets differ from those of `pools`, or a group has fewer than two
        models.
    """
    cached = set(cache.assets["asset"])
    if cached != set(pools.index):
        raise ValueError(
            f"the cache's assets differ from the universe's: only in the cache "
            f"{sorted(cached - set(pools.index))}, only in the universe "
            f"{sorted(set(pools.index) - cached)}"
        )
    every = dict(zip(cache.models["model"], cache.models["every_hours"], strict=True))
    groups = {"hourly": [model for model in models if int(every[model]) == 0]}
    if any(int(every[model]) > 0 for model in models):
        groups["grid"] = models
    windows = list(settings.windows)
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for group, members in groups.items():
        if len(members) < 2:
            raise ValueError(f"group {group} has fewer than two models: {members}")
        losses = common_losses(cache, members, windows=windows, horizon=settings.horizon)
        parts.append(
            _group_tables(losses, members, pools, group=group, splits=settings.splits, seed=seed)
        )
    kinds = dict(zip(cache.models["model"], cache.models["kind"], strict=True))
    scores = pd.concat([part[0] for part in parts], ignore_index=True)
    return {
        "ranking_scores": scores.assign(kind=scores["model"].map(kinds)),
        "ranking_parts": pd.concat([part[1] for part in parts], ignore_index=True),
        "ranking_splits": pd.concat([part[2] for part in parts], ignore_index=True),
    }


def stability_answer(splits: pd.DataFrame, settings: RankingSettings) -> tuple[str, float]:
    """Return the Q3 answer, ``yes`` or ``no``, and the median split tau of the hourly group.

    Parameters
    ----------
    splits : pd.DataFrame
        ``ranking_splits`` of `ranking_tables`.
    settings : RankingSettings
        The threshold.

    Returns
    -------
    tuple of (str, float)
        ``"yes"`` when the median tau of the hourly group is at least the threshold, else
        ``"no"``; and that median.

    Raises
    ------
    ValueError
        If the hourly group has no split.
    """
    median, _, _ = split_summary(splits, "hourly", level=settings.range_level)
    return ("yes" if median >= settings.tau_threshold else "no"), median
