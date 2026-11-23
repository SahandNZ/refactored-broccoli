"""Q1: every model on the decisions of the grid model (Kronos), and how many follow a closure."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.benchmark import (
    Bootstrap,
    ForecastCache,
    cell_stream,
    decision_losses,
    decision_points,
    scored_decisions,
)
from src.stats import circular_block_indices, has_enough_blocks, politis_white_block
from src.utils import closures

KEYS = ["time", "asset", "horizon"]
"""The columns that name one decision of a model (the cache's join keys)."""


@dataclass(frozen=True, slots=True)
class GridCache(ForecastCache):
    """A forecast cache whose zero-shot models forecast only the decisions of one grid model.

    Every zero-shot model's forecasts are cut to the (time, asset, horizon) keys the grid
    model forecast, and every model decides on the grid model's grid (``every_hours``), so
    `src.benchmark.null_tables` scores each model on the same decision points as the grid
    model, with the same timing and blocks. The baselines keep every forecast, so the best
    baseline is chosen on all development decisions as in rq1, and it is paired with each
    model on that model's decisions only.

    Attributes
    ----------
    keys : pd.DataFrame
        The grid model's decision keys, ``time``, ``asset`` and ``horizon``.
    """

    keys: pd.DataFrame

    def forecasts(self, model: str) -> pd.DataFrame:
        """Return `model`'s forecasts, on the grid model's decisions only for a zero-shot model.

        Parameters
        ----------
        model : str
            A model of the cache.

        Returns
        -------
        pd.DataFrame
            The rows of `ForecastCache.forecasts` (a zero-shot model: only those whose keys the
        grid model forecast).
        """
        frame = ForecastCache.forecasts(self, model)
        kinds = dict(zip(self.models["model"], self.models["kind"], strict=True))
        if kinds[model] == "baseline":
            return frame
        return frame.merge(self.keys, on=KEYS, how="inner", validate="many_to_one")


def grid_cache(cache: ForecastCache, grid_model: str) -> GridCache:
    """Return `cache` seen through the decisions and the grid of `grid_model`.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    grid_model : str
        A model of the cache that decides on a grid (``every_hours`` above 0).

    Returns
    -------
    GridCache
        The cache with every model on the grid model's decisions and grid.

    Raises
    ------
    ValueError
        If `grid_model` is not in the cache or does not decide on a grid.
    """
    grids = dict(zip(cache.models["model"], cache.models["every_hours"], strict=True))
    if grid_model not in grids or int(grids[grid_model]) <= 0:
        raise ValueError(f"{grid_model} is not a grid model of {cache.folder}: {grids}")
    keys = cache.forecasts(grid_model)[KEYS].drop_duplicates().reset_index(drop=True)
    models = cache.models.assign(every_hours=int(grids[grid_model]))
    base = replace(cache, models=models)
    return GridCache(base.folder, base.labels, base.models, base.assets, keys)


def reopening_shares(
    cache: ForecastCache, grid_model: str, *, windows: list[str], min_closure_hours: float
) -> pd.DataFrame:
    """Return, per market, the shares of decisions next to a closure: after it, and across it.

    A closure is at least `min_closure_hours` hours without an own bar (`src.utils.closures`,
    the closure finder of 08-closures). A decision (horizon 1, one per own bar) follows a
    closure when it is made at the close of the first bar after it, so the closure gap lies in
    the last bar of its input; its target spans a closure when it is made at the last close
    before it, so its target is the reopening bar (08-closures' first target). Both shares are
    counted over all labelled decisions of `windows` and over the grid model's decisions.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    grid_model : str
        The grid model (Kronos).
    windows : list of str
        The scored windows.
    min_closure_hours : float
        The shortest closure that counts, in hours without a bar.

    Returns
    -------
    pd.DataFrame
        ``market``, ``decisions``, ``reopening_share`` and ``target_closure_share`` (all
        decisions), ``grid_decisions``, ``grid_reopening_share`` and
        ``grid_target_closure_share``, one row per market.
    """
    labels = cache.labels[cache.labels["horizon"] == 1][["time", "asset", "window"]]
    parts: list[pd.DataFrame] = []
    for _, asset_labels in labels.groupby("asset", sort=True):
        ordered = asset_labels.sort_values("time").reset_index(drop=True)
        found = closures(
            pd.DatetimeIndex(ordered["time"]),
            bar=pd.Timedelta(hours=1),
            min_hours=min_closure_hours,
        )
        positions = found["position"].to_numpy(dtype=int)
        is_reopening = np.zeros(len(ordered), dtype=bool)
        is_reopening[positions] = True
        is_across = np.zeros(len(ordered), dtype=bool)
        is_across[positions - 1] = True
        parts.append(ordered.assign(is_reopening=is_reopening, is_across=is_across))
    decided = pd.concat(parts, ignore_index=True)
    decided = decided[decided["window"].isin(windows)]
    decided = decided.merge(cache.assets[["asset", "market"]], on="asset", validate="many_to_one")
    grid = cache.forecasts(grid_model)
    grid = grid.loc[grid["horizon"] == 1, ["time", "asset"]].drop_duplicates()
    on_grid = decided.merge(grid, on=["time", "asset"], how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for market, part in decided.groupby("market", sort=True):
        chosen = on_grid[on_grid["market"] == market]
        empty = chosen.empty
        rows.append(
            {
                "market": market,
                "decisions": len(part),
                "reopening_share": float(part["is_reopening"].mean()),
                "target_closure_share": float(part["is_across"].mean()),
                "grid_decisions": len(chosen),
                "grid_reopening_share": np.nan if empty else float(chosen["is_reopening"].mean()),
                "grid_target_closure_share": np.nan if empty else float(chosen["is_across"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _point_losses(cache: GridCache, model: str, *, windows: list[str], horizon: int) -> pd.Series:
    # one model's CRPS averaged over each market's assets at each decision point (as rq1)
    decisions, _ = scored_decisions(cache, model, windows=windows)
    losses = decision_losses(decisions[decisions["horizon"] == horizon])
    losses = losses.assign(point=decision_points(losses, cache, model))
    return losses.groupby(["market", "point"])["crps"].mean()


def kronos_gaps(
    cache: GridCache,
    models: list[str],
    grid_models: list[str],
    *,
    windows: list[str],
    horizon: int,
    bootstrap: Bootstrap,
) -> pd.DataFrame:
    """Return each hourly model's CRPS gain over each grid model on the same decision points.

    Per market, both models' losses are averaged over the market's assets at each decision
    point (`decision_points`, as rq1), on the points both scored; the gain is
    ``100 (1 - CRPS_model / CRPS_grid)`` (positive: the hourly model has the lower CRPS) with
    a circular-block bootstrap interval of the paired points (Politis-White block of the
    difference, at least 2; `cell_stream` per pair). A pair with fewer than
    `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks has no interval.

    Parameters
    ----------
    cache : GridCache
        The cache on the grid model's decisions (`grid_cache`).
    models : list of str
        The hourly zero-shot models.
    grid_models : list of str
        The grid models (Kronos-small, Kronos-base).
    windows : list of str
        The scored windows.
    horizon : int
        The horizon.
    bootstrap : Bootstrap
        rq1's bootstrap: its draws (``n_resamples``), ``level`` and ``seed``.

    Returns
    -------
    pd.DataFrame
        ``model``, ``grid_model``, ``market``, ``points``, ``gain``, ``gain_low`` and
        ``gain_high``, one row per pair and market.
    """
    losses = {
        model: _point_losses(cache, model, windows=windows, horizon=horizon)
        for model in [*models, *grid_models]
    }
    rows: list[dict[str, object]] = []
    for model in models:
        for grid in grid_models:
            paired = pd.concat({"model": losses[model], "grid": losses[grid]}, axis=1).dropna()
            for market, cell in paired.groupby(level="market", sort=True):
                rng = cell_stream(bootstrap.seed, "kronos-gap", model, grid, market, horizon)
                gain = _gain(cell["model"].to_numpy(), cell["grid"].to_numpy(), rng, bootstrap)
                rows.append({"model": model, "grid_model": grid, "market": market} | gain)
    columns = ["model", "grid_model", "market", "points", "gain", "gain_low", "gain_high"]
    return pd.DataFrame(rows, columns=columns)


def _gain(
    model: np.ndarray, grid: np.ndarray, rng: np.random.Generator, bootstrap: Bootstrap
) -> dict[str, object]:
    # the gain of the model over the grid model and its circular-block bootstrap interval
    gain = 100 * (1 - model.mean() / grid.mean())
    block = max(politis_white_block(model - grid), 2)
    if not has_enough_blocks(len(model), block):
        return {"points": len(model), "gain": gain, "gain_low": np.nan, "gain_high": np.nan}
    rows = circular_block_indices(len(model), rng, n_resamples=bootstrap.n_resamples, block=block)
    draws = 100 * (1 - model[rows].mean(axis=1) / grid[rows].mean(axis=1))
    tail = (1 - bootstrap.level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return {"points": len(model), "gain": gain, "gain_low": float(low), "gain_high": float(high)}
