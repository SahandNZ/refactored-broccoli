"""The decisions a scoring study (03-06, 09) scores, and the cells it reports.

Every study reads a model's forecasts the same way (`scored_decisions`): only valid decisions
(`valid_quantiles`), one forecast per decision (the seeds' quantiles averaged), with the
market of the asset and, for a zero-shot model, only after its release. Every study reports
every (market, horizon) cell of every model; a cell that is not scored says why
(`complete_cells`, `cell_timing`).
"""

import logging
import math
import zlib

import numpy as np
import pandas as pd

from src.benchmark._cache import KEYS, QUANTILE_COLUMNS, ForecastCache
from src.benchmark._settings import Bootstrap
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS
from src.utils import spacing_hours

logger = logging.getLogger(__name__)

DECISION = [*KEYS, "window", "label_time", "market"]
"""The columns that name one decision in a table of scores (``label_time`` rides along)."""

CELL = ["market", "horizon"]
"""The columns that name one cell of a model."""

SCORED = "scored"
"""The ``status`` of a scored cell; any other status starts with ``"not scored: "``."""

RELEASED_AFTER = "not scored: released after the scored folds"
NO_VALID_DECISION = "not scored: no valid decision"

ALL_FOLDS = (*DEVELOPMENT_FOLDS, *TEST_FOLDS, *LEAKAGE_FOLDS)


def valid_quantiles(quantiles: np.ndarray) -> np.ndarray:
    """Return which forecasts are valid: nine finite quantiles that never decrease.

    Parameters
    ----------
    quantiles : np.ndarray
        Shape ``(n, 9)``.

    Returns
    -------
    np.ndarray
        Booleans, shape ``(n,)``; equal neighbours are allowed.
    """
    is_ordered = (np.diff(quantiles, axis=1) >= 0).all(axis=1)
    return np.isfinite(quantiles).all(axis=1) & is_ordered


def split_models(
    cache: ForecastCache, models: list[str], *, windows: list[str]
) -> tuple[list[str], list[str]]:
    """Check that `models` and `windows` are in the cache and split the baselines off.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        Model names.
    windows : list of str
        Every window the study reads.

    Returns
    -------
    tuple of (list of str, list of str)
        The baselines and the zero-shot models of `models`, in their order.

    Raises
    ------
    ValueError
        If a model or a window is not in the cache.
    """
    kinds = cache.models.set_index("model")["kind"]
    unknown = sorted(set(models) - set(kinds.index))
    missing = sorted(set(windows) - set(cache.labels["window"]))
    if unknown or missing:
        raise ValueError(f"not in {cache.folder}: models {unknown}, windows {missing}")
    baselines = [name for name in models if kinds[name] == "baseline"]
    return baselines, [name for name in models if name not in baselines]


def release_date(cache: ForecastCache, model: str) -> pd.Timestamp:
    """Return the release date of `model` (00:00 UTC), NaT for a baseline."""
    releases = dict(zip(cache.models["model"], cache.models["release_date"], strict=True))
    return releases[model]


def is_after_release(frame: pd.DataFrame, release: pd.Timestamp) -> np.ndarray:
    """Return which rows were decided after `release`; every row when it is NaT (a baseline).

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with a ``time`` column (the decision time).
    release : pd.Timestamp
        The release date (UTC) or NaT.

    Returns
    -------
    np.ndarray
        Booleans, shape ``(len(frame),)``.
    """
    if pd.isna(release):
        return np.ones(len(frame), dtype=bool)
    return (frame["time"] > release).to_numpy()


def valid_decisions(
    cache: ForecastCache, model: str, *, windows: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the valid decisions of `model` in `windows`, one forecast per decision.

    See `scored_decisions`; this one keeps the decisions before the release too (rq4).
    """
    frame = cache.forecasts(model)
    return _one_forecast_per_decision(cache, model, frame[frame["window"].isin(windows)])


def scored_decisions(
    cache: ForecastCache, model: str, *, windows: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the decisions of `model` a study scores, one forecast per decision.

    A decision (time, asset, horizon) is scored when it is in `windows`, was made after the
    model's release (zero-shot models; `is_after_release`) and every seed's forecast is
    valid (`valid_quantiles`, the rule of 02-forecasts Q1); other decisions are left out and
    the invalid ones are counted. A sampling model's seeds are combined by averaging their
    quantiles level by level (Vincentization), so every study scores one forecast per
    decision.

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    model : str
        A model of the cache.
    windows : list of str
        The scored windows.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        The decisions: the label columns of the cache, ``QUANTILE_COLUMNS`` and ``market``,
        one row per decision. The invalid decisions per cell: ``market``, ``horizon``,
        ``invalid``.

    Raises
    ------
    ValueError
        If `model` is not in the cache or an asset has no market.
    """
    frame = cache.forecasts(model)
    keep = frame["window"].isin(windows) & is_after_release(frame, release_date(cache, model))
    return _one_forecast_per_decision(cache, model, frame[keep])


def _one_forecast_per_decision(
    cache: ForecastCache, model: str, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # a decision is invalid when any of its seeds' forecasts is; the valid ones get the mean of
    # their seeds' quantiles and their label columns back
    frame = frame.assign(is_valid=valid_quantiles(frame[list(QUANTILE_COLUMNS)].to_numpy(float)))
    is_valid = frame.groupby(list(KEYS))["is_valid"].transform("all").to_numpy()
    valid = frame.loc[is_valid, [*KEYS, *QUANTILE_COLUMNS]]
    quantiles = valid.groupby(list(KEYS), as_index=False).mean()
    decisions = cache.labels.merge(quantiles, on=list(KEYS), validate="one_to_one")
    invalid = add_market(frame.loc[~is_valid, list(KEYS)].drop_duplicates(), cache.assets, model)
    counts = invalid.groupby(CELL).size().rename("invalid").reset_index()
    if len(invalid) > 0:
        logger.warning("%s: %d invalid decisions left out", model, len(invalid))
    return add_market(decisions, cache.assets, model), counts


def cell_stream(seed: int, *names: object) -> np.random.Generator:
    """Return the random stream of one cell: `seed` and a hash of the cell's names.

    The stream depends on the cell itself (e.g. model, market, horizon), not on its place in a
    loop, so adding a model or leaving a cell out never changes another cell's draws, and two
    studies that draw for the same cell draw the same numbers.

    Parameters
    ----------
    seed : int
        The study's seed.
    *names : object
        What names the cell, e.g. ``("kronos-small", "crypto", 1)``.

    Returns
    -------
    np.random.Generator
        The cell's generator.
    """
    key = zlib.crc32("|".join(str(name) for name in names).encode())
    return np.random.default_rng([seed, key])


def decision_points(frame: pd.DataFrame, cache: ForecastCache, model: str) -> pd.Series:
    """Return the decision point of each row: its time, or its grid point for a grid model.

    A model that decides on a grid (Kronos, ``every_hours`` in ``models.csv``) takes at each
    grid point every asset's first decision at or after it, so the assets of a market that
    closes decide at different times of one grid point; those decisions are one point of the
    series a test runs on, the grid point, not several independent times. The grid starts at
    the window's start (`src.evaluate` folds), as in 02-forecasts.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``time`` and ``window``.
    cache : ForecastCache
        The forecast cache (``every_hours`` per model).
    model : str
        The model.

    Returns
    -------
    pd.Series
        UTC times, aligned with `frame`.
    """
    grids = dict(zip(cache.models["model"], cache.models["every_hours"], strict=True))
    every = int(grids[model])
    if every == 0:
        return frame["time"]
    starts = {fold.name: fold.evaluation_start for fold in ALL_FOLDS}
    start = frame["window"].map(starts)
    step = pd.Timedelta(hours=every)
    return start + ((frame["time"] - start) // step) * step


def add_market(frame: pd.DataFrame, assets: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return `frame` with the ``market`` of each row's asset as its last column.

    Raises
    ------
    ValueError
        If an asset is not in `assets` (the cache's ``assets.csv``), naming `model`.
    """
    frame = frame.merge(assets[["asset", "market"]], on="asset", how="left")
    if frame["market"].isna().any():
        missing = sorted(frame.loc[frame["market"].isna(), "asset"].unique())
        raise ValueError(f"assets {missing} of {model} are not in assets.csv")
    return frame


TIMING_COLUMNS = ["times", "spacing_hours", "overlap", "dm_lags", "block", "status"]
"""The columns of `cell_timing`."""


def cell_timing(decisions: pd.DataFrame, bootstrap: Bootstrap) -> dict[str, float | int | str]:
    """Return the decision times of one cell, how their labels overlap, its lags and block.

    The overlap of a decision is the number of later decision times of the cell before its
    label time: the decisions whose labels share bars with it. It is counted directly, so it
    is right for a cell that pools markets with different sessions too (a stock market with
    exchanges in Tokyo, Frankfurt and New York). The Diebold-Mariano lags are 1 plus the median
    overlap (rounded up): the horizon for hourly decisions of a market that trades around the
    clock, 1 for Kronos. The bootstrap block is the largest of the lags,
    ``ceil(block_hours / spacing)`` and ``min_block`` (`Bootstrap.block_length`): long enough
    to keep the overlap and about a day of autocorrelation.

    Parameters
    ----------
    decisions : pd.DataFrame
        The cell's decisions (any number of assets): ``time`` and ``label_time``.
    bootstrap : Bootstrap
        The study's bootstrap.

    Returns
    -------
    dict
        ``times`` (distinct decision times), ``spacing_hours`` (their median gap; inf with
        fewer than two), ``overlap`` (the median overlap, rounded up), ``dm_lags`` (1 plus
        it), ``block`` and ``status``: `SCORED`, or why not when the cell has fewer times
        than its block.
    """
    times = pd.DatetimeIndex(decisions["time"].drop_duplicates().sort_values())
    later = times.searchsorted(pd.DatetimeIndex(decisions["label_time"])) - times.searchsorted(
        pd.DatetimeIndex(decisions["time"]), side="right"
    )
    overlap = math.ceil(float(np.median(later))) if len(decisions) > 0 else 0
    lags = 1 + overlap
    spacing = spacing_hours(times) if len(times) > 1 else math.inf
    block = max(lags, bootstrap.block_length(spacing))
    status = SCORED
    if len(times) < block:
        status = f"not scored: {len(times)} decision times, fewer than the bootstrap block {block}"
    return {
        "times": len(times),
        "spacing_hours": spacing,
        "overlap": overlap,
        "dm_lags": lags,
        "block": block,
        "status": status,
    }


def cell_timings(decisions: pd.DataFrame, bootstrap: Bootstrap, *, by: list[str]) -> pd.DataFrame:
    """Return `cell_timing` of every group of `decisions` (e.g. ``by=CELL``), one row each.

    Parameters
    ----------
    decisions : pd.DataFrame
        Decisions with the columns of `by`, ``time`` and ``label_time``.
    bootstrap : Bootstrap
        The study's bootstrap.
    by : list of str
        The columns that name a group.

    Returns
    -------
    pd.DataFrame
        The columns of `by` and ``TIMING_COLUMNS``; empty (with those columns) without
        decisions.
    """
    rows = [
        dict(zip(by, key, strict=True)) | cell_timing(group, bootstrap)
        for key, group in decisions.groupby(by)
    ]
    return pd.DataFrame(rows, columns=[*by, *TIMING_COLUMNS])


def complete_cells(
    cells: pd.DataFrame,
    cache: ForecastCache,
    model: str,
    *,
    invalid: pd.DataFrame,
    windows: list[str],
) -> pd.DataFrame:
    """Add the cells of `model` without a scored decision, and the invalid count of every cell.

    Every market of the cache and every horizon of its labels is a cell of every model. A
    cell missing from `cells` gets a row with ``times = 0`` and the ``status``
    `RELEASED_AFTER` when the model was released after the last decision of `windows`, else
    `NO_VALID_DECISION`.

    Parameters
    ----------
    cells : pd.DataFrame
        One row per cell of the model with decisions, with ``market``, ``horizon`` and
        ``status``; may be empty.
    cache : ForecastCache
        The forecast cache.
    model : str
        The model.
    invalid : pd.DataFrame
        The invalid counts of `scored_decisions`.
    windows : list of str
        The scored windows.

    Returns
    -------
    pd.DataFrame
        `cells` plus the missing cells, with ``model`` and ``invalid``, sorted by cell.
    """
    expected = pd.MultiIndex.from_product(
        [sorted(cache.assets["market"].unique()), sorted(cache.labels["horizon"].unique())],
        names=CELL,
    ).to_frame(index=False)
    last = cache.labels.loc[cache.labels["window"].isin(windows), "time"].max()
    release = release_date(cache, model)
    reason = RELEASED_AFTER if pd.notna(release) and release >= last else NO_VALID_DECISION
    known = [] if cells.empty else list(zip(cells["market"], cells["horizon"], strict=True))
    absent = expected[~pd.MultiIndex.from_frame(expected).isin(known)]
    absent = absent.assign(times=0, status=reason)
    if len(absent) > 0:
        logger.warning("%s: %d cells %s", model, len(absent), reason)
    table = pd.concat([part for part in (cells, absent) if not part.empty], ignore_index=True)
    table = table.merge(invalid, on=CELL, how="left").fillna({"invalid": 0})
    table["invalid"] = table["invalid"].astype(int)
    return table.assign(model=model).sort_values(CELL, ignore_index=True)


def require_scored(cells: pd.DataFrame, *, study: str, windows: list[str]) -> None:
    """Raise when no cell of the zero-shot models of a study could be scored.

    Parameters
    ----------
    cells : pd.DataFrame
        The cells of the study's zero-shot models, with ``status``.
    study : str
        The study, for the message (e.g. ``"rq1"``).
    windows : list of str
        The scored windows, for the message.

    Raises
    ------
    ValueError
        If no cell is scored, with the count of every reason.
    """
    if (cells["status"] == SCORED).any():
        return
    counts = cells.groupby("status").size()
    reasons = "; ".join(f"{count} cells {status}" for status, count in counts.items())
    raise ValueError(f"no zero-shot model of {study} could be scored on {windows}: {reasons}")


def unscored_note(cells: pd.DataFrame) -> str:
    """Return the sentence of a results README that lists the cells not scored, or ``""``."""
    unscored = cells[cells["status"] != SCORED]
    if unscored.empty:
        return ""
    counts = unscored.groupby("status").size()
    reasons = "; ".join(f"{count} {status}" for status, count in counts.items())
    return f" Cells not scored ({len(unscored)}, in the table with their status): {reasons}."
