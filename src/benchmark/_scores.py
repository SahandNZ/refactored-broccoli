"""CRPS of the cached forecasts and the comparison of each zero-shot model with the best baseline.

Shared by the benchmark studies: `decision_losses` scores decisions and `best_baseline` gives
the losses of the baseline chosen per market and horizon. rq1 adds its tables (`null_tables`)
and the test of its decision rule (`beats_baseline`), which the study's verdict and its power
analysis (`src.benchmark._power`) apply.
"""

import numpy as np
import pandas as pd
from scipy.special import stdtrit

from src.benchmark._cache import QUANTILE_COLUMNS, ForecastCache
from src.benchmark._decisions import (
    CELL,
    DECISION,
    SCORED,
    cell_stream,
    cell_timings,
    complete_cells,
    decision_points,
    release_date,
    require_scored,
    scored_decisions,
    split_models,
)
from src.benchmark._settings import NullRule, StudySettings
from src.evaluate import Fold, quantile_crps
from src.models import QUANTILE_LEVELS
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    circular_block_indices,
    diebold_mariano,
    has_enough_blocks,
    politis_white_block,
)


def decision_losses(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return the CRPS of every decision of `scored_decisions`.

    The CRPS of a forecast is twice the mean pinball loss over its nine quantiles
    (`src.evaluate.quantile_crps`).

    Parameters
    ----------
    decisions : pd.DataFrame
        Decisions with ``DECISION``, ``label`` and ``QUANTILE_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        ``DECISION`` and ``crps``, one row per decision.
    """
    quantiles = decisions[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    labels = decisions["label"].to_numpy(dtype=float)
    crps = quantile_crps(quantiles, np.asarray(QUANTILE_LEVELS), labels)
    return decisions[DECISION].assign(crps=crps)


def baseline_losses(
    cache: ForecastCache, baselines: list[str], *, windows: list[str]
) -> dict[str, pd.DataFrame]:
    """Return the `decision_losses` of every baseline on its decisions in `windows`."""
    return {
        name: decision_losses(scored_decisions(cache, name, windows=windows)[0])
        for name in baselines
    }


def best_baseline(
    losses: dict[str, pd.DataFrame], *, choice_windows: list[str], scored_windows: list[str]
) -> pd.DataFrame:
    """Return, per scored window, market and horizon, the losses of the best baseline.

    For each scored window the best baseline of a cell is the one with the lowest CRPS on
    `choice_windows` other than that window, on the decisions every baseline forecast (so all
    baselines are judged on the same decisions), averaged as the comparison averages: over the
    market's assets at each decision time, then over the times. With the development folds as
    both choice and scored windows, each fold's baseline is chosen on the other three (leave
    one fold out); with the test folds scored, every fold's baseline is chosen on d1-d4. A
    choice decision made before a scored window whose label ends inside it is never used.

    Parameters
    ----------
    losses : dict of str to pd.DataFrame
        For each baseline, its `decision_losses`.
    choice_windows, scored_windows : list of str
        The windows the choice is made on (the development folds) and the scored windows.

    Returns
    -------
    pd.DataFrame
        The columns of `decision_losses` plus ``baseline``, the name of the chosen baseline;
        one row per decision of that baseline in `scored_windows`.

    Raises
    ------
    ValueError
        If a cell of a scored window has no decision in its choice windows that every
        baseline forecast.
    """
    stacked = pd.concat(
        [frame.assign(baseline=name) for name, frame in losses.items()], ignore_index=True
    )
    is_shared = stacked.groupby(DECISION)["baseline"].transform("size") == len(losses)
    parts: list[pd.DataFrame] = []
    for window in scored_windows:
        others = [name for name in choice_windows if name != window]
        scored = stacked[stacked["window"] == window]
        start = pd.Timestamp(scored["time"].min())
        reaches_in = (stacked["time"] < start) & (stacked["label_time"] > start)
        is_choice = stacked["window"].isin(others) & is_shared & ~reaches_in
        choice = stacked[is_choice]
        per_time = choice.groupby([*CELL, "baseline", "time"])["crps"].mean()
        means = per_time.groupby([*CELL, "baseline"]).mean().reset_index()
        best = means.loc[means.groupby(CELL)["crps"].idxmin(), [*CELL, "baseline"]]
        chosen = scored.merge(best, on=[*CELL, "baseline"])
        n_cells, n_chosen = len(scored[CELL].drop_duplicates()), len(chosen[CELL].drop_duplicates())
        if n_chosen < n_cells:
            raise ValueError(
                f"only {n_chosen} of {n_cells} market and horizon cells of {window} have "
                f"decisions in {others} that every baseline {sorted(losses)} forecast"
            )
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def baseline_names(values: pd.Series) -> str:
    """Return the names of the chosen baselines of a cell, sorted and joined by commas.

    While the plan is open each development fold may have its own best baseline.
    """
    return ", ".join(sorted(set(values)))


def _cell_test(
    cell: pd.DataFrame, overlap: int, *, rule: NullRule, rng: np.random.Generator, level: float
) -> dict[str, float | int | str]:
    # the rq1 test of one cell on its per-point series: the Politis-White block (at least
    # BLOCK_OVERLAP_FLOOR times the labels' overlap), the one-sided Diebold-Mariano test and
    # the skill with its bootstrap interval; a cell with fewer than 5 blocks is untested
    model = cell["crps_model"].to_numpy(dtype=float)
    baseline = cell["crps_baseline"].to_numpy(dtype=float)
    block = max(politis_white_block(model - baseline), BLOCK_OVERLAP_FLOOR * overlap, 2)
    skill = 100 * (1 - model.mean() / baseline.mean())
    if not has_enough_blocks(len(cell), block):
        return {
            "block": block,
            "status": f"untested: {len(cell)} decision times, fewer than "
            f"{MIN_BOOTSTRAP_BLOCKS} blocks of {block}",
            "dm_lags": np.nan,
            "dm_statistic": np.nan,
            "p_value": np.nan,
            "skill": skill,
            "skill_low": np.nan,
            "skill_high": np.nan,
        }
    test = diebold_mariano(model, baseline, overlap=overlap)
    indices = circular_block_indices(len(cell), rng, n_resamples=rule.skill_resamples, block=block)
    # the interval is skill +- t(n // block - 1) x the circular-bootstrap standard error of
    # the skill, the method of rq2 and rq4 (src.stats.bootstrap_mean_test)
    draws = 100 * (1 - model[indices].mean(axis=1) / baseline[indices].mean(axis=1))
    half_width = float(stdtrit(len(cell) // block - 1, 1 - (1 - level) / 2)) * float(
        draws.std(ddof=1)
    )
    return {
        "block": block,
        "status": SCORED,
        "dm_lags": test["lags"],
        "dm_statistic": test["statistic"],
        "p_value": test["p_less"],
        "skill": skill,
        "skill_low": skill - half_width,
        "skill_high": skill + half_width,
    }


def _cells(
    times: pd.DataFrame, model: str, *, rule: NullRule, seed: int, level: float
) -> pd.DataFrame:
    # Diebold-Mariano per cell on the loss averaged over the market's assets at each decision
    # point (the assets of one market move together, so they are not independent tests); the
    # cell's mean CRPS, its skill and its quarters are unweighted means over the same points.
    # Each cell draws its skill interval from its own stream (`cell_stream`)
    rows: list[dict[str, object]] = []
    for (market, horizon), cell in times.groupby(CELL):
        timing = cell.iloc[0]
        by_window = cell.groupby("window")[["crps_model", "crps_baseline"]].mean()
        rng = cell_stream(seed, "skill", model, market, horizon)
        test = _cell_test(cell, int(timing["overlap"]), rule=rule, rng=rng, level=level)
        rows.append(
            {
                "market": market,
                "horizon": horizon,
                "baseline": baseline_names(cell["baseline"]),
                "decisions": int(cell["decisions"].sum()),
                "times": int(timing["times"]),
                "spacing_hours": timing["spacing_hours"],
                "overlap": int(timing["overlap"]),
                "model_crps": cell["crps_model"].mean(),
                "baseline_crps": cell["crps_baseline"].mean(),
                **test,
                "quarters": len(by_window),
                "winning_quarters": int(
                    (by_window["crps_model"] < by_window["crps_baseline"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def full_quarters(release: pd.Timestamp, folds: tuple[Fold, ...]) -> int:
    """Return how many of `folds` start at or after `release`: quarters wholly after it."""
    return sum(fold.evaluation_start >= release for fold in folds)


def null_tables(
    cache: ForecastCache, models: list[str], settings: StudySettings, rule: NullRule
) -> dict[str, pd.DataFrame]:
    """Compare every zero-shot model of `models` with the best baseline (rq1).

    The best baseline of each scored fold, market and horizon is chosen on the development
    folds (`best_baseline`). A model is scored on its `scored_decisions` in
    ``settings.scored_folds`` and the baseline on exactly those decisions (a decision the
    baseline did not forecast is left out of both). The losses are averaged over the market's
    assets at each decision point (`decision_points`: the decision time, or for Kronos the
    grid point, whose assets decide at different times when their market closes); the cell's
    CRPS, its skill, its winning quarters and its Diebold-Mariano test all use that series,
    unweighted. The test (`src.stats.diebold_mariano`, one-sided: H1 the model has the lower
    CRPS) gets the median overlap of the cell's labels (`cell_timing`); the block of the
    circular-block bootstrap of the skill interval and of the power analysis is the
    Politis-White block of the loss difference, at least `BLOCK_OVERLAP_FLOOR` (4) times the
    overlap. A cell with fewer than five blocks of decision points
    (`src.stats.has_enough_blocks`) is untested, and a cell without any decision is not
    scored (`status`). Each model belongs to the ``main`` Holm family when at least
    ``rule.main_family_min_quarters`` scored folds start after its scoring start, else to the
    ``late`` family.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The zero-shot models and the baselines to compare (their kind comes from the cache).
    settings : StudySettings
        The folds and the bootstrap of the study.
    rule : NullRule
        The skill draws and the family rule of rq1.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"times"``: one row per scored model, market, horizon and decision point, in time
        order, with ``baseline``, ``window``, ``decisions``, the means over the market's
        assets ``crps_model`` and ``crps_baseline`` (the series of the test and of the power
        analysis), the cell's ``overlap``, ``block`` and ``family``. ``"quarters"``: one row
        per model, market, horizon and window with ``baseline``, ``decisions``,
        ``model_crps`` and ``baseline_crps``. ``"cells"``: one row per model, market and
        horizon (every cell, `status` says whether it is scored) with the same means over all
        windows, ``skill`` (``100 (1 - model_crps / baseline_crps)``) with ``skill_low`` and
        ``skill_high`` (the bootstrap interval at ``settings.bootstrap.level``), ``times``,
        ``spacing_hours``, ``overlap``, ``block``, ``dm_lags``, ``invalid``, the
        Diebold-Mariano ``dm_statistic`` (negative when the model is better) and its
        one-sided ``p_value``, ``quarters``, ``winning_quarters``, ``full_quarters`` and
        ``family``.

    Raises
    ------
    ValueError
        If a model or a fold is not in the cache, or no cell of a zero-shot model is scored.
    """
    choice = [fold.name for fold in settings.choice_folds]
    scored = [fold.name for fold in settings.scored_folds]
    baselines, zero_shot = split_models(cache, models, windows=choice + scored)
    losses = baseline_losses(cache, baselines, windows=sorted(set(choice + scored)))
    best = best_baseline(losses, choice_windows=choice, scored_windows=scored)
    tables: dict[str, list[pd.DataFrame]] = {"times": [], "quarters": [], "cells": []}
    for name in zero_shot:
        decisions, invalid = scored_decisions(cache, name, windows=scored)
        paired = decision_losses(decisions).merge(
            best, on=DECISION, suffixes=("_model", "_baseline")
        )
        paired["time"] = decision_points(paired, cache, name)
        times = paired.groupby([*CELL, "window", "time"], as_index=False).agg(
            baseline=("baseline", "first"),
            decisions=("crps_model", "size"),
            crps_model=("crps_model", "mean"),
            crps_baseline=("crps_baseline", "mean"),
            label_time=("label_time", "max"),
        )
        timings = cell_timings(times, settings.bootstrap, by=CELL)
        times = times.merge(timings[[*CELL, "times", "spacing_hours", "overlap"]], on=CELL)
        cells = _cells(
            times, name, rule=rule, seed=settings.bootstrap.seed, level=settings.bootstrap.level
        )
        quarters_after = full_quarters(release_date(cache, name), settings.scored_folds)
        family = "main" if quarters_after >= rule.main_family_min_quarters else "late"
        cells = complete_cells(cells, cache, name, invalid=invalid, windows=scored)
        cells = cells.assign(full_quarters=quarters_after, family=family)
        quarters = times.groupby([*CELL, "window"], as_index=False).agg(
            baseline=("baseline", "first"),
            decisions=("decisions", "sum"),
            model_crps=("crps_model", "mean"),
            baseline_crps=("crps_baseline", "mean"),
        )
        if not times.empty:
            timing = cells[[*CELL, "block", "status"]]
            dropped = times.drop(columns=["label_time", "times", "spacing_hours"])
            with_timing = dropped.merge(timing, on=CELL)
            is_scored = with_timing[with_timing["status"] == SCORED]
            tables["times"].append(
                is_scored.sort_values([*CELL, "time"]).assign(model=name, family=family)
            )
        tables["quarters"].append(quarters.assign(model=name))
        tables["cells"].append(cells)
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    require_scored(result["cells"], study="rq1", windows=scored)
    return result


def beats_baseline(
    adjusted_p: np.ndarray,
    winning_quarters: np.ndarray,
    dm_statistic: np.ndarray,
    *,
    alpha: float,
    min_winning_quarters: int,
) -> np.ndarray:
    """Return which cells beat the best baseline under the rule of rq1.

    A cell beats the baseline if it wins at least `min_winning_quarters` windows, its adjusted
    Diebold-Mariano p-value is below `alpha` and the statistic favours the model (negative).
    A cell without a test (NaN) never does.

    Parameters
    ----------
    adjusted_p, winning_quarters, dm_statistic : np.ndarray
        Per cell: the multiple-testing adjusted one-sided p-value, the windows won and the
        Diebold-Mariano statistic, each of shape ``(n,)``.
    alpha : float
        The significance level, e.g. 0.05.
    min_winning_quarters : int
        The fewest windows the model must win, e.g. 3 of the 4 test quarters.

    Returns
    -------
    np.ndarray
        Booleans, shape ``(n,)``.
    """
    return (winning_quarters >= min_winning_quarters) & (adjusted_p < alpha) & (dm_statistic < 0)
