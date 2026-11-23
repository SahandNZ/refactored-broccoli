"""Study 07-rq5-off-hours, Q1: score the reopening forecasts and test the arms against each other.

`reopening_tables` scores every arm on the weekends of the scored folds, pools a market's
assets once per weekend and tests each pair of `COMPARISONS` with a one-sided
Diebold-Mariano test (H1: the arm has the lower CRPS than its reference), Holm-adjusted over
every test of every cell, each with a moving-block bootstrap interval of the CRPS skill.
`covariate_use` measures how much of the crypto move a model passes on from its covariate.
"""

import math

import numpy as np
import pandas as pd
from scipy.special import ndtri

from src.benchmark import QUANTILE_COLUMNS, SCORED, valid_quantiles
from src.evaluate import quantile_crps
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    block_indices,
    diebold_mariano,
    has_enough_blocks,
    holm,
    politis_white_block,
)

from ._reopening import ARMS, LEVELS
from ._settings import OffHoursSettings

COMPARISONS = (
    ("covariate", "calendar"),
    ("covariate", "event"),
    ("covariate", "correction"),
    ("covariate", "model-free"),
    ("correction", "event"),
    ("model-free", "event"),
)
"""Each test of a cell as (arm, reference): the covariate against the same calendar hours
without it (its own effect), the benchmark's event time, the correction and the model-free
forecast; and both crypto-based arms without a covariate against event time."""

REFERENCE = "ewma-normal"
"""The benchmark's naive reference: the standard normal in label units (the EWMA-normal
forecast), reported next to the arms and not tested."""

NORMAL_QUANTILES = ndtri(LEVELS)
"""Its nine quantiles, shape ``(9,)``."""

COVARIATE_TESTS = COMPARISONS[:4]
"""The tests a cell's covariate arm must all pass to be the best way to feed the model."""

CELL = ["market", "model"]
"""The columns that name one cell of Q1: a market's assets are pooled once per weekend."""

TEST_COLUMNS = [
    *CELL,
    "arm",
    "reference",
    "times",
    "crps_arm",
    "crps_reference",
    "skill",
    "skill_low",
    "skill_high",
    "block",
    "dm_lags",
    "dm_statistic",
    "p_value",
    "status",
]
"""The columns of a test of `reopening_tables` before ``adjusted_p`` and ``is_better``."""


def label_overlap(times: pd.DatetimeIndex, label_times: pd.DatetimeIndex) -> int:
    """Return the median number of later decision times before each label is known, rounded up.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Sorted decision times, shape ``(n,)``.
    label_times : pd.DatetimeIndex
        When each decision's label is known, shape ``(n,)``.

    Returns
    -------
    int
        The overlap (0 when every label is known before the next decision).
    """
    later = times.searchsorted(label_times) - times.searchsorted(times, side="right")
    return math.ceil(float(np.median(later))) if len(times) else 0


def _scored(forecasts: pd.DataFrame, settings: OffHoursSettings) -> pd.DataFrame:
    # the forecasts of the scored folds with their CRPS and 10-90 % coverage, NaN where the
    # forecast is not valid or the label is unknown
    scored = forecasts[forecasts["fold"] != ""]
    if scored.empty:
        raise ValueError(
            "no forecast of a weekend in the scored folds "
            f"{[fold.name for fold in settings.scored_folds]}: check split_date and the data"
        )
    quantiles = scored[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    labels = scored["label"].to_numpy(dtype=float)
    is_valid = valid_quantiles(quantiles) & np.isfinite(labels)
    crps = quantile_crps(np.where(is_valid[:, None], quantiles, 0.0), LEVELS, labels)
    covered = (labels >= quantiles[:, 0]) & (labels <= quantiles[:, -1])
    known = np.where(np.isfinite(labels), labels, 0.0)
    normal = quantile_crps(np.tile(NORMAL_QUANTILES, (len(scored), 1)), LEVELS, known)
    normal_covered = np.abs(known) <= NORMAL_QUANTILES[-1]
    return scored.assign(
        crps=np.where(is_valid, crps, np.nan),
        covered=np.where(is_valid, covered, np.nan),
        reference_crps=normal,
        reference_covered=normal_covered.astype(float),
    )


def _test(
    arm: np.ndarray,
    reference: np.ndarray,
    overlap: int,
    rng: np.random.Generator,
    settings: OffHoursSettings,
) -> dict[str, float | int | str]:
    # one-sided Diebold-Mariano (H1: the arm's loss is lower) and the block-bootstrap interval
    # of the skill, 100 (1 - CRPS arm / CRPS reference); the block is the Politis-White block
    # of the loss difference, at least BLOCK_OVERLAP_FLOOR times the label overlap. A series
    # with fewer than MIN_BOOTSTRAP_BLOCKS blocks is untested: no p-value, no interval
    block = min(
        max(politis_white_block(arm - reference), BLOCK_OVERLAP_FLOOR * overlap, 1), len(arm)
    )
    result: dict[str, float | int | str] = {
        "times": len(arm),
        "crps_arm": float(arm.mean()),
        "crps_reference": float(reference.mean()),
        "skill": 100 * (1 - float(arm.mean()) / float(reference.mean())),
        "skill_low": np.nan,
        "skill_high": np.nan,
        "block": block,
        "dm_lags": np.nan,
        "dm_statistic": np.nan,
        "p_value": np.nan,
        "status": SCORED,
    }
    if not has_enough_blocks(len(arm), block):
        result["status"] = (
            f"not tested: {len(arm)} weekly points, fewer than {MIN_BOOTSTRAP_BLOCKS} "
            f"blocks of {block}"
        )
        return result
    test = diebold_mariano(arm, reference, overlap=overlap)
    indices = block_indices(len(arm), rng, n_resamples=settings.bootstrap_resamples, block=block)
    draws = 100 * (1 - arm[indices].mean(axis=1) / reference[indices].mean(axis=1))
    tail = (1 - settings.interval_level) / 2
    return result | {
        "skill_low": float(np.quantile(draws, tail)),
        "skill_high": float(np.quantile(draws, 1 - tail)),
        "dm_lags": int(test["lags"]),
        "dm_statistic": test["statistic"],
        "p_value": test["p_less"],
    }


def _weekly(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    # one market and model: the asset-weekends valid in every arm, their CRPS and coverage
    # averaged over the assets once per weekend (`week`), however staggered their reopenings;
    # the overlap of those weekly points, and the number of asset-weekends kept
    index = ["asset", "week"]
    crps = group.pivot(index=index, columns="arm", values="crps").reindex(columns=list(ARMS))
    covered = group.pivot(index=index, columns="arm", values="covered")
    is_paired = crps.notna().all(axis=1)
    timing = group.drop_duplicates(index).set_index(index).loc[is_paired[is_paired].index]
    crps = crps[is_paired].assign(**{REFERENCE: timing["reference_crps"]})
    covered = covered.reindex(columns=list(ARMS))[is_paired]
    covered = covered.assign(**{REFERENCE: timing["reference_covered"]})
    paired = crps.groupby(level="week").mean()
    coverage = covered.groupby(level="week").mean()
    times = timing.groupby(level="week").agg({"reopen_time": "min", "label_time": "max"})
    overlap = label_overlap(
        pd.DatetimeIndex(times["reopen_time"]), pd.DatetimeIndex(times["label_time"])
    )
    return paired, coverage, overlap, int(is_paired.sum())


def _width_at_edge(group: pd.DataFrame, settings: OffHoursSettings) -> float:
    # the share of the cell's scored corrections whose width is at either end of the grid,
    # where the grid may have bound the fit
    widths = group.loc[(group["arm"] == "correction") & group["crps"].notna(), "width"]
    values: np.ndarray = widths.to_numpy(dtype=float)
    lowest, highest = settings.width_factors[0], settings.width_factors[-1]
    is_edge: np.ndarray = np.isclose(values, lowest) | np.isclose(values, highest)
    return float(is_edge.mean()) if len(values) else np.nan


def reopening_tables(
    forecasts: pd.DataFrame, settings: OffHoursSettings
) -> dict[str, pd.DataFrame]:
    """Score every market and model on the weekends of the scored folds and test Q1.

    An asset-weekend is scored when all five arms are valid (nine finite, ordered quantiles)
    and its label is known; the others are counted in ``excluded``. A market's assets are
    averaged once per weekend (the ``week`` of the Friday close), so assets that reopen at
    different hours give one point. Each pair of `COMPARISONS` gets a one-sided
    Diebold-Mariano test on that weekly series (overlap of its labels, `label_overlap`;
    `src.stats.diebold_mariano`: Newey-West lags at least the overlap and Andrews' choice,
    Student's t with ``n - 1`` degrees of freedom) and a moving-block bootstrap interval of
    the skill; all p-values of all cells are Holm-adjusted together. A cell with fewer than
    ``min_weekends`` weekly points is not tested.

    Parameters
    ----------
    forecasts : pd.DataFrame
        `reopening_forecasts`.
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    dict of str to pd.DataFrame
        ``reopening_cells``: one row per cell (`CELL`): ``weekends`` (asset-weekends scored),
        ``times`` (weekly points), ``excluded``, ``overlap``, ``crps_<arm>`` and
        ``coverage_<arm>`` (share of labels inside the 10-90 % interval) of every arm and of
        `REFERENCE` (on the same weekly points, not tested), ``width_at_edge`` (the share of
        the scored corrections whose width is at an end of ``width_factors``), ``status`` and
        ``covariate_is_best`` (every test of `COVARIATE_TESTS` passes).
        ``reopening_tests``: one row per tested cell and comparison: `TEST_COLUMNS`,
        ``adjusted_p`` and ``is_better`` (adjusted p below ``alpha``). A comparison whose
        weekly series holds fewer than `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks is not tested
        (``status`` says why): no p-value, no interval, outside the Holm family.

    Raises
    ------
    ValueError
        If no forecast falls in the scored folds.
    """
    scored = _scored(forecasts, settings)
    cells: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    for (market, model), group in scored.groupby(CELL, sort=True):
        paired, coverage, overlap, kept = _weekly(group)
        cell: dict[str, object] = {
            "market": market,
            "model": model,
            "weekends": kept,
            "times": len(paired),
            "excluded": group.drop_duplicates(["asset", "week"]).shape[0] - kept,
            "overlap": overlap,
        }
        cell |= {f"crps_{arm}": float(paired[arm].mean()) for arm in (*ARMS, REFERENCE)}
        cell |= {f"coverage_{arm}": float(coverage[arm].mean()) for arm in (*ARMS, REFERENCE)}
        cell["width_at_edge"] = _width_at_edge(group, settings)
        cell["status"] = SCORED
        if len(paired) < settings.min_weekends:
            cell["status"] = (
                f"not tested: {len(paired)} weekends, fewer than {settings.min_weekends}"
            )
        cells.append(cell)
        if cell["status"] != SCORED:
            continue
        for arm, reference in COMPARISONS:
            rng = np.random.default_rng([settings.bootstrap_seed, len(tests)])
            values = _test(
                paired[arm].to_numpy(), paired[reference].to_numpy(), overlap, rng, settings
            )
            test: dict[str, object] = {"market": market, "model": model, "arm": arm}
            test |= {"reference": reference, **values}
            tests.append(test)
    test_table = pd.DataFrame(tests, columns=TEST_COLUMNS)
    test_table["adjusted_p"] = holm(test_table["p_value"].to_numpy(dtype=float))
    test_table["is_better"] = test_table["adjusted_p"] < settings.alpha
    return {
        "reopening_cells": _with_verdict(pd.DataFrame(cells), test_table),
        "reopening_tests": test_table,
    }


def _with_verdict(cells: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    # a cell's covariate arm is the best way to feed the model when it beats every other arm
    pairs = list(zip(tests["arm"], tests["reference"], strict=True))
    is_covariate_test = np.array([pair in COVARIATE_TESTS for pair in pairs], dtype=bool)
    covariate = tests[is_covariate_test]
    passed = covariate.groupby(CELL)["is_better"].agg(["all", "size"])
    best = passed["all"] & (passed["size"] == len(COVARIATE_TESTS))
    keys = pd.MultiIndex.from_frame(cells[CELL])
    return cells.assign(covariate_is_best=best.reindex(keys, fill_value=False).to_numpy())


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    # the least-squares slope with an intercept: centred cross product over centred square
    centred = x - x.mean()
    return float(centred @ (y - y.mean()) / (centred @ centred))


def covariate_use(
    forecasts: pd.DataFrame, gap_assets: pd.DataFrame, settings: OffHoursSettings
) -> pd.DataFrame:
    """Describe how much of the weekend crypto move each model passes on from its covariate.

    On the asset-weekends Q1 scores (all five arms valid, so the crypto move, the move and
    sigma are known), ``use_slope`` is the slope (least squares with an intercept) of the
    difference of the medians of the covariate and the calendar arm, back in log-return units
    (times sigma), on the weekend crypto move;
    ``move_slope`` is the slope of the realised move on the crypto move over the same
    asset-weekends, so ``use_share = use_slope / move_slope`` is near 0 when a model ignores
    its covariate and near 1 when it passes on as much of the move as the data hold.
    ``gap_beta`` is the median Q2 coefficient (a partial slope of the gap, with Friday's
    information, over all Q2 weekends) of the market's tested forecast assets, shown next to
    it. A descriptive number: it has no interval and decides nothing.

    Parameters
    ----------
    forecasts : pd.DataFrame
        `reopening_forecasts`.
    gap_assets : pd.DataFrame
        The ``gap_assets`` table of `gap_tables`.
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    pd.DataFrame
        One row per market and model: ``weekends``, ``use_slope``, ``move_slope``,
        ``use_share`` and ``gap_beta``.
    """
    scored = _scored(forecasts, settings)
    median = QUANTILE_COLUMNS[len(QUANTILE_COLUMNS) // 2]
    index = ["market", "model", "asset", "week"]
    valid = scored.pivot(index=index, columns="arm", values="crps").reindex(columns=list(ARMS))
    is_scored = valid.notna().all(axis=1)
    medians = scored.pivot(index=index, columns="arm", values=median)
    medians = medians.loc[is_scored[is_scored].index]
    rows = scored.drop_duplicates(index).set_index(index).loc[medians.index]
    shift = (medians["covariate"] - medians["calendar"]) * rows["sigma"]
    frame = pd.DataFrame(
        {"shift": shift, "crypto": rows["weekend_crypto"], "move": rows["move"]}
    ).reset_index()
    tested = gap_assets[
        (gap_assets["status"] == "tested") & gap_assets["asset"].isin(scored["asset"])
    ]
    medians_by_market = tested.groupby("market")["beta"].median()
    betas = {str(market): float(beta) for market, beta in medians_by_market.items()}
    table: list[dict[str, object]] = []
    for (market, model), group in frame.groupby(["market", "model"], sort=True):
        crypto = group["crypto"].to_numpy(dtype=float)
        use = _slope(crypto, group["shift"].to_numpy(dtype=float))
        move = _slope(crypto, group["move"].to_numpy(dtype=float))
        table.append(
            {"market": market, "model": model, "weekends": len(group), "use_slope": use,
             "move_slope": move, "use_share": use / move if move else np.nan,
             "gap_beta": betas.get(str(market), np.nan)}
        )  # fmt: skip
    columns = ["market", "model", "weekends", "use_slope", "move_slope", "use_share", "gap_beta"]
    return pd.DataFrame(table, columns=columns)
