"""Skill of each zero-shot model before and after its release, against the best baseline (rq4).

`leakage_periods` puts each decision of a model into its period, `leakage_grid` keeps the
decisions of the grid of the leakage quarters, and `leakage_tables` gives the skill over the
best baseline per period and time and the tests of every model, market and horizon
(`_compare.leakage_cells`).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark import (
    CELL,
    DECISION,
    ForecastCache,
    StudySettings,
    baseline_losses,
    baseline_names,
    best_baseline,
    cell_timings,
    complete_cells,
    decision_losses,
    decision_points,
    is_after_release,
    release_date,
    require_scored,
    split_models,
    valid_decisions,
)
from src.evaluate import LEAKAGE_FOLDS, Fold
from src.experiment import StudyConfig
from src.utils import to_utc

from ._compare import TESTS, leakage_cells


@dataclass(frozen=True, slots=True)
class LeakageOptions:
    """The options of rq4 beyond the shared settings of a scoring study.

    Attributes
    ----------
    grid_hours : int
        The hours between the decisions of the leakage quarters (``step_hours.leakage`` of
        02-forecasts); every period of a model that decides every hour keeps the decisions of
        this grid (`leakage_grid`), so all periods have the same spacing.
    placebo_models : tuple of str
        The models of the placebo test.
    placebo_split : pd.Timestamp
        The decision time (UTC) that splits the placebo models' before period in two.
    not_built : tuple of str
        Models whose leakage quarters are not built (owner's choice): with the leakage
        quarters as the before period, every cell of theirs is reported with the status
        `NOT_BUILT` and no test (and a forecast of theirs in l1-l12 raises).
    """

    grid_hours: int
    placebo_models: tuple[str, ...]
    placebo_split: pd.Timestamp
    not_built: tuple[str, ...]


NOT_BUILT = "not scored: not built on the leakage quarters"
"""The ``status`` of every cell of a model in ``LeakageOptions.not_built``."""


def leakage_options(config: StudyConfig) -> LeakageOptions:
    """Read the options of rq4 beyond the shared settings of a scoring study.

    The options are ``grid_hours``, ``placebo_models``, ``placebo_split`` and ``not_built``.

    Raises
    ------
    KeyError
        If an option is missing (there are no defaults).
    """
    options: dict[str, Any] = config.options
    return LeakageOptions(
        grid_hours=int(options["grid_hours"]),
        placebo_models=tuple(str(name) for name in options["placebo_models"]),
        placebo_split=to_utc(str(options["placebo_split"])),
        not_built=tuple(str(name) for name in options["not_built"]),
    )


def leakage_periods(
    losses: pd.DataFrame,
    *,
    release: pd.Timestamp,
    cutoff: pd.Timestamp,
    before_windows: list[str],
    between_windows: list[str],
    after_windows: list[str],
) -> pd.DataFrame:
    """Put every decision of one model into its period: before, between or after its release.

    A decision's label time is the close time of its label's last bar (``label_time``).

    - ``before``: a decision in `before_windows` whose label time is earlier than the
      pretraining cutoff and the release (00:00 UTC of the dates): data the model may have seen.
    - ``between``: a decision in `between_windows` made at or after the cutoff whose label
      time is earlier than the release; only when the model has a cutoff.
    - ``after``: a decision in `after_windows` made after the release (`is_after_release`, as
      in every other study).

    Other decisions (a label across the cutoff or the release) are left out.

    Parameters
    ----------
    losses : pd.DataFrame
        The `decision_losses` of the model with ``label_time``.
    release : pd.Timestamp
        The model's release date (UTC).
    cutoff : pd.Timestamp
        The end of its pretraining data (UTC), or NaT when unknown.
    before_windows, between_windows, after_windows : list of str
        The windows of the before period (the leakage quarters), of the between period (the
        leakage quarters and the development folds, so a model released in 2025 has its
        months before the release) and of the after period (the scored folds).

    Returns
    -------
    pd.DataFrame
        The rows of `losses` in a period, with the column ``period``.

    Raises
    ------
    ValueError
        If `release` is NaT.
    """
    if pd.isna(release):
        raise ValueError("a zero-shot model needs a release date, got NaT")
    label_time = losses["label_time"]
    is_early = label_time < release
    seen_until = release if pd.isna(cutoff) else min(cutoff, release)
    in_before = losses["window"].isin(before_windows)
    in_between = losses["window"].isin(between_windows)
    period = pd.Series(None, index=losses.index, dtype="object")
    period[in_before & is_early & (label_time < seen_until)] = "before"
    period[in_between & is_early & (losses["time"] >= seen_until)] = "between"
    is_after = losses["window"].isin(after_windows) & is_after_release(losses, release)
    period[is_after] = "after"
    return losses.assign(period=period).dropna(subset=["period"])


def leakage_grid(labels: pd.DataFrame, *, hours: int, folds: tuple[Fold, ...]) -> pd.DataFrame:
    """Return the decisions that the grid of the leakage quarters would make, re-timed.

    The leakage quarters decide every `hours` from each window's start (00:00 UTC), and an
    asset decides at a grid point only when one of its bars closed since the previous point,
    with its last close as the context end (02-forecasts). With hourly decisions (one per own
    close) that decision is the asset's last decision in ``(point - hours, point]``: the same
    context, label and forecast. A window already on the grid (a leakage quarter) keeps every
    decision at its own time. A point at or after the window's end is left out.

    Parameters
    ----------
    labels : pd.DataFrame
        The cache's label rows, with ``time``, ``asset`` and ``window`` (a fold of `folds`);
        every horizon may have a row.
    hours : int
        Hours between grid points.
    folds : tuple of Fold
        The folds of the windows.

    Returns
    -------
    pd.DataFrame
        One row per kept decision: ``time`` (the hourly decision), ``asset``, ``window`` and
        ``point`` (its grid point).
    """
    keys = labels[["time", "asset", "window"]].drop_duplicates()
    starts = keys["window"].map({fold.name: fold.evaluation_start for fold in folds})
    ends = keys["window"].map({fold.name: fold.evaluation_end for fold in folds})
    steps = np.ceil((keys["time"] - starts) / pd.Timedelta(hours=hours))
    keys = keys.assign(point=starts + pd.to_timedelta(steps * hours, unit="h"))
    keys = keys[keys["point"] < ends]
    last = keys.groupby(["asset", "point"])["time"].transform("max")
    return keys[keys["time"] == last]


def _model_periods(
    paired: pd.DataFrame,
    *,
    grid: pd.DataFrame,
    every_hours: int,
    release: pd.Timestamp,
    cutoff: pd.Timestamp,
    windows: tuple[list[str], list[str], list[str]],
) -> pd.DataFrame:
    # the rows of every period a test of the model uses. A model that decides every hour is
    # first re-timed to the grid (the leakage quarters are on it already), then gets its
    # periods, so a decision has the period of its grid point; its hourly after period is
    # after-hourly. A model on its own coarse grid (Kronos) is at its points already.
    on_grid = paired
    parts: list[pd.DataFrame] = []
    if every_hours == 0:
        on_grid = paired.merge(grid, on=["time", "asset", "window"], validate="many_to_one")
        on_grid = on_grid.drop(columns="time").rename(columns={"point": "time"})
        hourly = leakage_periods(
            paired, release=release, cutoff=cutoff, before_windows=windows[0],
            between_windows=windows[1], after_windows=windows[2],
        )  # fmt: skip
        parts.append(hourly[hourly["period"] == "after"].assign(period="after-hourly"))
    periods = leakage_periods(
        on_grid, release=release, cutoff=cutoff, before_windows=windows[0],
        between_windows=windows[1], after_windows=windows[2],
    )  # fmt: skip
    parts.append(periods)
    return pd.concat(parts, ignore_index=True)


def _placebo_periods(periods: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    # the periods of a placebo model plus early and late: its before period split at `split`
    before = periods[periods["period"] == "before"]
    is_early = before["time"] < split
    early = before[is_early].assign(period="early")
    return pd.concat([periods, early, before[~is_early].assign(period="late")], ignore_index=True)


def _model_tests(
    model: str, *, cutoff: pd.Timestamp, every_hours: int, options: LeakageOptions
) -> list[str]:
    # every model has the test leakage; hourly only a model that decides every hour (Kronos's
    # after period is on its own grid in both), cutoff and placebo only where they apply
    tests = ["leakage"]
    if every_hours == 0:
        tests.append("hourly")
    if pd.notna(cutoff):
        tests.append("cutoff")
    if model in options.placebo_models:
        tests.append("placebo")
    return tests


def _model_tables(
    periods: pd.DataFrame, settings: StudySettings, *, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # the periods table and the per-time skills of one model: the skill is averaged over the
    # market's assets at each decision time, and the label overlap of each period is counted
    # on those times (the last label time of each), as in rq1
    period = periods.groupby([*CELL, "period"], as_index=False).agg(
        baseline=("baseline", baseline_names),
        decisions=("crps_model", "size"),
        model_crps=("crps_model", "mean"),
        baseline_crps=("crps_baseline", "mean"),
    )
    periods = periods.assign(skill=periods["crps_baseline"] - periods["crps_model"])
    by_time = periods.groupby([*CELL, "baseline", "period", "time"], as_index=False)
    times = by_time.agg(label_time=("label_time", "max"))
    timings = cell_timings(times, settings.bootstrap, by=[*CELL, "period"])
    overlaps = timings[[*CELL, "period", "overlap"]]
    skills = by_time["skill"].mean().merge(overlaps, on=[*CELL, "period"])
    return period.assign(model=model), skills.assign(model=model)


COUNTS = ["first_times", "second_times", "first_block", "second_block", "unpaired"]
"""The integer columns of a test's cells."""


def _unpaired(periods: pd.DataFrame, test: str) -> pd.DataFrame:
    # per cell, the decisions of the test's two periods without a valid forecast of the
    # cell's control (e.g. a baseline fit that did not converge, listed by 02-forecasts):
    # they cannot be paired and leave the cell; counted so the table shows them
    names = TESTS[test]
    rows = periods[(periods["control"] == "left_only") & periods["period"].isin(names)]
    return rows.groupby(CELL).size().rename("unpaired").reset_index()


def _complete_test(cells: pd.DataFrame, *, test: str) -> pd.DataFrame:
    # the cells of one test with every cell of `complete_cells`: its periods named, the
    # counts as integers (0 for a cell without decisions) and no column ``times``
    counts = cells.reindex(columns=COUNTS).fillna(0).astype(int)
    first, second = TESTS[test]
    cells = cells.drop(columns="times", errors="ignore").assign(test=test, first=first)
    return cells.assign(second=second, **{column: counts[column] for column in COUNTS})


def leakage_tables(
    cache: ForecastCache,
    models: list[str],
    settings: StudySettings,
    *,
    before_folds: tuple[Fold, ...],
    options: LeakageOptions,
) -> dict[str, pd.DataFrame]:
    """Skill of every zero-shot model of `models` over the best baseline, and its tests.

    The best baseline of each window, market and horizon is chosen on the development folds
    (`best_baseline`). Each valid decision of a model (one forecast per decision) goes into
    its period (`leakage_periods`; after = ``settings.scored_folds``; between also takes the
    development folds' decisions before the release, with their baseline chosen on the other
    development folds, so a model released in 2025 has its months between the cutoff and the
    release) and is paired with the best baseline's loss on the same decision (a decision the
    baseline did not forecast is left out). Every period but the hourly after period of the
    check ``hourly`` keeps the decisions the grid of the leakage quarters would make
    (`leakage_grid`); a model on its own grid (Kronos) counts the decisions of one grid point
    as one time (`decision_points`). The skill of a decision is the baseline's CRPS minus the
    model's (positive: the model is better); it is averaged over the market's assets at each
    time, because the assets of one market move together. Each test of `TESTS` that applies
    to the model gives one row per cell (`leakage_cells`).

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The zero-shot models and the baselines (their kind comes from the cache).
    settings : StudySettings
        The folds of the after period, the choice of the baseline and the bootstrap.
    before_folds : tuple of Fold
        The folds of the before period (the leakage quarters l1-l12); the between period uses
        them and ``settings.choice_folds``.
    options : LeakageOptions
        The grid of the periods, the placebo test and the models not built on the leakage
        quarters.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"periods"``: one row per model, market, horizon and period with ``baseline``,
        ``decisions``, ``model_crps`` and ``baseline_crps``. ``"skills"``: one row per model,
        market, horizon, period and time with ``baseline``, ``skill`` and the period's label
        ``overlap``. ``"cells"``: one row per test, model, market and horizon
        (`leakage_cells`, plus ``test``, ``first``, ``second``, ``invalid`` and
        ``unpaired``: the decisions of the test's two periods without a valid forecast of the
        cell's control, left out of the cell).

    Raises
    ------
    ValueError
        If a model or a window is not in the cache, a placebo or not-built model is not a
        zero-shot model of `models`, a not-built model has a forecast (valid or not) in the
        leakage quarters, or a placebo model is not a zero-shot model
        of `models`, or no cell of the test ``leakage`` is scored.
    """
    choice = [fold.name for fold in settings.choice_folds]
    before = [fold.name for fold in before_folds]
    between = sorted(set(before + choice))
    after = [fold.name for fold in settings.scored_folds]
    scored = sorted(set(between + after))
    baselines, zero_shot = split_models(cache, models, windows=choice + scored)
    for listed, kind in ((options.placebo_models, "placebo"), (options.not_built, "not built")):
        stray = sorted(set(listed) - set(zero_shot))
        if stray:
            raise ValueError(f"{kind} models {stray} are not zero-shot models of {models}")
    is_leakage = {fold.name for fold in before_folds} == {fold.name for fold in LEAKAGE_FOLDS}
    losses = baseline_losses(cache, baselines, windows=sorted(set(choice + scored)))
    best = best_baseline(losses, choice_windows=choice, scored_windows=scored)
    folds = (*before_folds, *settings.choice_folds, *settings.scored_folds)
    labels = cache.labels[cache.labels["window"].isin(scored)]
    grid = leakage_grid(labels, hours=options.grid_hours, folds=folds)
    tables: dict[str, list[pd.DataFrame]] = {"periods": [], "skills": [], "cells": []}
    cutoffs = cache.models.set_index("model")["pretraining_cutoff"]
    every_hours = dict(zip(cache.models["model"], cache.models["every_hours"], strict=True))
    for name in zero_shot:
        decisions, invalid = valid_decisions(cache, name, windows=scored)
        is_not_built = is_leakage and name in options.not_built
        if is_not_built and cache.forecasts(name)["window"].isin(before).any():
            raise ValueError(f"{name} is listed as not built but has forecasts in {before}")
        # every decision gets its period, paired with the control or not ("control"), so the
        # decisions without a control forecast are counted per period (`_unpaired`)
        marked = decision_losses(decisions).merge(
            best, on=DECISION, how="left", suffixes=("_model", "_baseline"),
            validate="one_to_one", indicator="control",
        )  # fmt: skip
        marked = marked.assign(time=decision_points(marked, cache, name))
        marked = _model_periods(
            marked, grid=grid, every_hours=int(every_hours[name]),
            release=release_date(cache, name), cutoff=cutoffs[name],
            windows=(before, between, after),
        )  # fmt: skip
        if name in options.placebo_models:
            marked = _placebo_periods(marked, options.placebo_split)
        paired = marked[marked["control"] == "both"].drop(columns="control")
        period, skills = _model_tables(paired, settings, model=name)
        tables["periods"].append(period)
        tables["skills"].append(skills)
        tests = _model_tests(
            name, cutoff=cutoffs[name], every_hours=int(every_hours[name]), options=options
        )
        for test in tests:
            cells = leakage_cells(skills, test=test, bootstrap=settings.bootstrap)
            cells = complete_cells(cells, cache, name, invalid=invalid, windows=after)
            if is_not_built:
                cells = cells.assign(status=NOT_BUILT, did=np.nan, low=np.nan, high=np.nan)
                cells = cells.assign(p_value=np.nan)
            cells = cells.merge(_unpaired(marked, test), on=CELL, how="left")
            tables["cells"].append(_complete_test(cells, test=test))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    require_scored(
        result["cells"][result["cells"]["test"] == "leakage"], study="rq4", windows=scored
    )
    return result
