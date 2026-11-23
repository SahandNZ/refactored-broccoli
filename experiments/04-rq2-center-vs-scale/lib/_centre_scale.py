"""Centre and scale of the cached forecasts (rq2): the sign of the median and the calibration.

`centre_scale_tables` scores every model per market, horizon and window; `centre_scale_verdict`
applies the decision rule of rq2.

- Centre: the hit rate of the sign of the median against that of the centre reference
  (climatology) on the same decisions, its rank IC with the label, and the CRPS of the
  forecast and of the forecast moved so that its median is zero (``centred_crps``: what is
  left is only the scale); ``centre_share`` is the part of the centred CRPS the median removes.
- Scale: ``below_qXX``, the share of labels at or below each of the nine quantiles (the
  PIT's distribution function at the nine levels); the coverage of the 10-90 % interval
  (``below_q90 - below_q10``) and the Kolmogorov-Smirnov distance of the PIT at the nine
  levels, ``max |below_qXX - level|``.
"""

import math
import zlib

import numpy as np
import pandas as pd

from src.benchmark import (
    CELL,
    DECISION,
    QUANTILE_COLUMNS,
    SCORED,
    Bootstrap,
    ForecastCache,
    StudySettings,
    cell_timing,
    complete_cells,
    decision_points,
    full_quarters,
    release_date,
    require_scored,
    scored_decisions,
    split_models,
)
from src.evaluate import Fold, quantile_crps, rank_ic
from src.models import QUANTILE_LEVELS
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    bootstrap_mean_test,
    has_enough_blocks,
    holm,
    is_constant,
    politis_white_block,
)

from ._bootstrap import ratio_interval
from ._settings import CentreScaleRule

BELOW = [f"below_{column}" for column in QUANTILE_COLUMNS]
"""Per decision, whether the label is at or below each quantile; per table, the share."""

REFERENCE_BELOW = [f"reference_{column}" for column in BELOW]
"""The same for the scale reference, on the same decisions."""

TARGET_COVERAGE = QUANTILE_LEVELS[-1] - QUANTILE_LEVELS[0]
"""The coverage of a calibrated 10-90 % interval, 0.8."""

TEST_COLUMNS = [
    "hit_low",
    "hit_high",
    "diff_low",
    "diff_high",
    "p_value",
    "coverage_low",
    "coverage_high",
]
"""The bootstrap columns of a cell: the intervals and the one-sided p-value."""

FLOAT_SLACK = 1e-9
"""Added to the coverage tolerance, so a coverage of exactly 0.75 or 0.85 (an error that is
0.05 in exact arithmetic but 0.05 +- 1e-16 in floats) is judged the same on both sides."""

TESTED = "tested"
"""The ``centre_test`` of a cell whose centre was tested; otherwise it says why not."""

MAIN, LATE = "main", "late"
"""The two Holm families: models with at least ``main_family_min_quarters`` scored quarters
wholly after their scoring start, and the others."""


def _decision_scores(decisions: pd.DataFrame) -> pd.DataFrame:
    # the CRPS of the forecast and of the forecast with its median moved to zero, and whether
    # the label is at or below each quantile, for every decision of `scored_decisions`
    quantiles = decisions[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    label = decisions["label"].to_numpy(dtype=float)
    median = decisions["q50"].to_numpy(dtype=float)
    levels = np.asarray(QUANTILE_LEVELS)
    crps = quantile_crps(quantiles, levels, label)
    centred = quantile_crps(quantiles - median[:, None], levels, label)
    below = pd.DataFrame(label[:, None] <= quantiles, columns=BELOW)
    scores = decisions[[*DECISION, "label"]].assign(median=median, crps=crps, centred_crps=centred)
    return pd.concat([scores.reset_index(drop=True), below], axis=1)


def _reference_scores(
    cache: ForecastCache, rule: CentreScaleRule, windows: list[str]
) -> pd.DataFrame:
    # the scale reference's shares below each quantile and the centre reference's median, on
    # the decisions both forecast
    scale = _decision_scores(scored_decisions(cache, rule.scale_reference, windows=windows)[0])
    scale = scale[[*DECISION, *BELOW]].rename(
        columns=dict(zip(BELOW, REFERENCE_BELOW, strict=True))
    )
    centre = scored_decisions(cache, rule.centre_reference, windows=windows)[0]
    centre = centre[DECISION].assign(reference_median=centre["q50"].to_numpy(dtype=float))
    return scale.merge(centre, on=DECISION, validate="one_to_one")


def _add_signs(paired: pd.DataFrame) -> pd.DataFrame:
    # a decision has a sign only when the label and both medians are non-zero; its hit is
    # whether the model's median has the label's sign, its reference hit the same for the
    # centre reference; `inside` is whether the label is in the 10-90 % interval
    label = paired["label"].to_numpy()
    median = paired["median"].to_numpy()
    reference = paired["reference_median"].to_numpy()
    has_sign = (label != 0) & (median != 0) & (reference != 0)
    hit = np.where(has_sign, np.sign(median) == np.sign(label), np.nan)
    reference_hit = np.where(has_sign, np.sign(reference) == np.sign(label), np.nan)
    inside = paired[BELOW[-1]] & ~paired[BELOW[0]]
    return paired.assign(hit=hit, reference_hit=reference_hit, inside=inside)


def _per_time(group: pd.DataFrame) -> pd.DataFrame:
    # every decision point weighs the same: the means over the market's assets at each point
    # (the assets of one market move together, as in the test of rq1), in time order; a
    # point's label time is its latest, for the overlap count of `cell_timing`
    times = group.groupby("time", as_index=False).agg(
        label_time=("label_time", "max"),
        decisions=("label", "size"),
        inside=("inside", "sum"),
        hit=("hit", "mean"),
        reference_hit=("reference_hit", "mean"),
        crps=("crps", "mean"),
        centred_crps=("centred_crps", "mean"),
    )
    return times.assign(hit_difference=times["hit"] - times["reference_hit"])


def _summary(group: pd.DataFrame) -> dict[str, float]:
    # hit rates and CRPS are means over decision times; the PIT shares, the coverage and the
    # KS distance are plain means over decisions
    times = _per_time(group)
    shares = group[BELOW].mean().to_numpy()
    reference = group[REFERENCE_BELOW].mean().to_numpy()
    both = np.vstack([shares, reference])
    coverage_error = np.abs(both[:, -1] - both[:, 0] - TARGET_COVERAGE)
    ks_distance = np.abs(both - np.asarray(QUANTILE_LEVELS)).max(axis=1)
    crps, centred = float(times["crps"].mean()), float(times["centred_crps"].mean())
    summary = {
        "decisions": len(group),
        "times": len(times),
        "signed_decisions": int(group["hit"].notna().sum()),
        "zero_reference_share": float((group["reference_median"] == 0).mean()),
        "signed_times": int(times["hit_difference"].notna().sum()),
        "hit_rate": times["hit"].mean(),
        "reference_hit_rate": times["reference_hit"].mean(),
        "hit_difference": times["hit_difference"].mean(),
        "rank_ic": rank_ic(group["median"], group["label"]),
        "crps": crps,
        "centred_crps": centred,
        "centre_share": 1 - crps / centred if centred > 0 else math.nan,
        "coverage": shares[-1] - shares[0],
        "coverage_error": coverage_error[0],
        "ks_distance": ks_distance[0],
        "reference_coverage_error": coverage_error[1],
        "reference_ks_distance": ks_distance[1],
    }
    return summary | dict(zip(BELOW, shares, strict=True))


def _quarters(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (market, horizon, window), group in paired.groupby([*CELL, "window"]):
        rows.append({"market": market, "horizon": horizon, "window": window, **_summary(group)})
    return pd.DataFrame(rows)


def _block(times: pd.DataFrame, overlap: int) -> int:
    # the block of every test and interval of the cell: Politis-White on the series the cell
    # tests (the hit difference per decision point), or on the coverage per point when the
    # cell has no signs, at least BLOCK_OVERLAP_FLOOR (4) times the overlap of its labels
    # (and 2), as in rq1
    differences = times["hit_difference"].dropna().to_numpy(dtype=float)
    coverage = (times["inside"] / times["decisions"]).to_numpy(dtype=float)
    series = differences if len(differences) > 0 else coverage
    return max(politis_white_block(series), BLOCK_OVERLAP_FLOOR * overlap, 2)


def _cell_tests(
    times: pd.DataFrame, block: int, bootstrap: Bootstrap, *, seed: int
) -> dict[str, object]:
    # the one-sided test of the hit difference and every interval of a scored cell, all
    # circular-block bootstraps with its one block and seed; each needs at least
    # MIN_BOOTSTRAP_BLOCKS blocks of its decision points (the coverage all of them, the
    # centre those with a sign), and a difference without spread has no test
    result: dict[str, object] = dict.fromkeys(TEST_COLUMNS, math.nan)
    level, draws = bootstrap.level, bootstrap.n_resamples
    if has_enough_blocks(len(times), block):
        result["coverage_low"], result["coverage_high"] = ratio_interval(
            times["inside"].to_numpy(dtype=float),
            times["decisions"].to_numpy(dtype=float),
            level=level, n_resamples=draws, seed=seed, block=block,
        )  # fmt: skip
    signed = times.dropna(subset=["hit_difference"])
    differences = signed["hit_difference"].to_numpy(dtype=float)
    if not has_enough_blocks(len(signed), block):
        result["centre_test"] = (
            f"untested: {len(signed)} decision times with a sign, fewer than "
            f"{MIN_BOOTSTRAP_BLOCKS} blocks of {block}"
        )
        return result
    test = bootstrap_mean_test(
        differences, null=0.0, level=level, n_resamples=draws, seed=seed, block=block
    )
    hits = bootstrap_mean_test(
        signed["hit"].to_numpy(dtype=float),
        null=0.0, level=level, n_resamples=draws, seed=seed, block=block,
    )  # fmt: skip
    result.update(
        hit_low=hits["low"], hit_high=hits["high"], diff_low=test["low"], diff_high=test["high"],
        p_value=test["p_greater"], centre_test=TESTED,
    )  # fmt: skip
    # a difference without spread keeps its intervals (a point) but has no test
    if is_constant(differences, scale=1.0):
        result.update(p_value=math.nan, centre_test="untested: no spread in the hit difference")
    return result


def cell_seed(bootstrap: Bootstrap, model: str, market: str, horizon: int) -> int:
    """Return the seed of the bootstrap draws of one cell, from its names only.

    An independent stream per model and cell that does not depend on which other models or
    cells a run holds (``--models``), so a cell's draws are the same in every run.
    """
    keys = [bootstrap.seed, zlib.crc32(model.encode()), zlib.crc32(market.encode()), horizon]
    return int(np.random.SeedSequence(keys).generate_state(1)[0])


def _cells(paired: pd.DataFrame, bootstrap: Bootstrap, *, model: str) -> pd.DataFrame:
    # a cell with fewer decision points than its block is not scored
    rows: list[dict[str, object]] = []
    for (market, horizon), cell in paired.groupby(CELL):
        times = _per_time(cell)
        timing = cell_timing(times, bootstrap)
        overlap = int(timing["overlap"])
        block = _block(times, overlap)
        status = SCORED
        tests: dict[str, object] = {**dict.fromkeys(TEST_COLUMNS, math.nan), "centre_test": ""}
        if len(times) < block:
            status = (
                f"not scored: {len(times)} decision times, fewer than the bootstrap block {block}"
            )
        else:
            seed = cell_seed(bootstrap, model, str(market), int(str(horizon)))
            tests = _cell_tests(times, block, bootstrap, seed=seed)
        rows.append(
            {
                "market": market,
                "horizon": horizon,
                "quarters": cell["window"].nunique(),
                "spacing_hours": timing["spacing_hours"],
                "overlap": overlap,
                "block": block,
                "status": status,
                **_summary(cell),
                **tests,
            }
        )
    return pd.DataFrame(rows)


def _full_quarters(release: pd.Timestamp, folds: tuple[Fold, ...]) -> int:
    # rq1's `full_quarters`; a baseline (NaT) has every quarter
    return len(folds) if pd.isna(release) else full_quarters(release, folds)


def _unpaired(decisions: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    # the decisions of each cell left out because a reference did not forecast them
    before = decisions.groupby(CELL).size()
    after = paired.groupby(CELL).size().reindex(before.index, fill_value=0)
    return (before - after).rename("unpaired").reset_index()


def centre_scale_tables(
    cache: ForecastCache, models: list[str], settings: StudySettings, *, rule: CentreScaleRule
) -> dict[str, pd.DataFrame]:
    """Score the centre and the scale of every model of `models` per market and horizon (rq2).

    Every model is scored on its `scored_decisions` in ``settings.scored_folds`` (a zero-shot
    model only after its release), one forecast per decision. The scale reference and the
    centre reference of `rule` are scored on exactly the same decisions; a decision either
    did not forecast is left out and counted (``unpaired``). The decisions are grouped by
    decision point (`decision_points`: the decision time, or for a grid model like Kronos the
    grid point, whose assets decide at different times when their market closes), as in rq1;
    "decision time" below means that point.

    Every interval and the test of a cell use one circular block, ``block``: the Politis-White
    block (`src.stats.politis_white_block`) of the series the cell tests, at least
    `src.stats.BLOCK_OVERLAP_FLOOR` (4) times the overlap of its labels (`cell_timing`) and
    at least 2, as in rq1. A cell with fewer
    decision times than its block, or without any decision, is not scored (``status``). The
    centre test is one-sided (H1: the model's hit rate is above the centre reference's), on
    the hit difference per decision time: `src.stats.bootstrap_mean_test`, a circular-block
    bootstrap standard error read against Student's t with ``n // block - 1`` degrees of
    freedom; the intervals are its t intervals (the coverage's from `ratio_interval`). A test
    or interval needs at least `src.stats.MIN_BOOTSTRAP_BLOCKS` (5) blocks of its decision
    times (the centre: those with a sign); a cell with fewer, or whose difference has no
    spread, has no test (NaN, outside Holm) and says why in ``centre_test`` (`TESTED`
    otherwise).

    Parameters
    ----------
    cache : ForecastCache
        The forecast cache.
    models : list of str
        The models to score (their kind comes from the cache); the scale reference is always
        scored too.
    settings : StudySettings
        The scored folds and the bootstrap.
    rule : CentreScaleRule
        The scale and the centre reference.

    Returns
    -------
    dict of str to pd.DataFrame
        ``"quarters"``: one row per model, market, horizon and window. ``"cells"``: one row
        per model, market and horizon over all windows (every cell; ``status`` says whether
        it is scored), plus ``release_date``, ``quarters``, ``full_quarters`` (scored folds
        that start at or after the scoring start), ``spacing_hours``, ``overlap``,
        ``block``, ``invalid``, ``unpaired``, ``centre_test`` and ``TEST_COLUMNS``; the
        cells' bootstrap draws use one seed per model and cell (`cell_seed`). Both have
        ``model``, ``kind``,
        ``decisions``, ``times``, ``signed_decisions``, ``signed_times``,
        ``zero_reference_share`` (the share of the decisions whose centre-reference median is
        exactly zero, which leave the centre test), ``hit_rate``,
        ``reference_hit_rate`` and ``hit_difference`` (means over decision times of the
        cross-asset share, on the decisions with a sign), ``rank_ic`` (pooled Spearman of the
        median with the label), ``crps``, ``centred_crps`` and ``centre_share`` (means over
        decision times), ``coverage``, ``coverage_error``, ``ks_distance``,
        ``reference_coverage_error``, ``reference_ks_distance`` and ``BELOW``.

    Raises
    ------
    ValueError
        If a model or a fold is not in the cache, `models` has no zero-shot model, or no cell
        of a zero-shot model is scored.
    """
    windows = [fold.name for fold in settings.scored_folds]
    references = [rule.scale_reference, rule.centre_reference]
    zero_shot_models = split_models(cache, [*models, *references], windows=windows)[1]
    if not zero_shot_models:
        raise ValueError(f"rq2 judges zero-shot models; {models} are all baselines")
    kinds = cache.models.set_index("model")["kind"]
    reference_scores = _reference_scores(cache, rule, windows)
    quarters: list[pd.DataFrame] = []
    cells: list[pd.DataFrame] = []
    # the scale reference always has its own rows: the figure shows it first
    for name in dict.fromkeys([*models, rule.scale_reference]):
        decisions, invalid = scored_decisions(cache, name, windows=windows)
        paired = _add_signs(_decision_scores(decisions).merge(reference_scores, on=DECISION))
        paired["time"] = decision_points(paired, cache, name)
        quarters.append(_quarters(paired).assign(model=name, kind=kinds[name]))
        cells_of_model = _cells(paired, settings.bootstrap, model=name)
        scored = complete_cells(cells_of_model, cache, name, invalid=invalid, windows=windows)
        scored = scored.merge(_unpaired(decisions, paired), on=CELL, how="left")
        scored = scored.fillna({"unpaired": 0}).astype({"unpaired": int})
        release = release_date(cache, name)
        cells.append(
            scored.assign(
                kind=kinds[name],
                release_date=release,
                full_quarters=_full_quarters(release, settings.scored_folds),
            )
        )
    tables = {
        "quarters": pd.concat(quarters, ignore_index=True),
        "cells": pd.concat(cells, ignore_index=True),
    }
    zero_shot = tables["cells"][tables["cells"]["kind"] == "zero-shot"]
    require_scored(zero_shot, study="rq2", windows=windows)
    return tables


def _family_holm(p_values: np.ndarray, families: np.ndarray) -> np.ndarray:
    # Holm within each family on its own; NaN (no test) stays NaN and is not counted
    adjusted = np.full(len(p_values), math.nan)
    for family in (MAIN, LATE):
        members = families == family
        adjusted[members] = holm(p_values[members])
    return adjusted


def centre_scale_verdict(
    cells: pd.DataFrame, quarters: pd.DataFrame, *, alpha: float, rule: CentreScaleRule
) -> pd.DataFrame:
    """Apply the decision rule of rq2 to the tables of `centre_scale_tables`.

    Only scored cells (``status``) can pass. The cells form two Holm families, each adjusted
    on its own and counting only its cells with a test: ``main``, the models with at least
    `main_family_min_quarters` scored quarters wholly after their scoring start
    (``full_quarters``), and ``late``, the others.

    - Centre skill (Q1): hit rate above the centre reference's (``hit_difference > 0``), the
      one-sided bootstrap p-value below `alpha` after Holm in the cell's family, and a
      positive rank IC in at least `min_quarters` windows. Worse centre (descriptive, not a
      test): the whole interval of the hit difference below 0.
    - Calibrated (Q2): coverage error at most `coverage_tolerance` in at least `min_quarters`
      windows.
    - Better scale than the reference (Q2): coverage error and KS distance both below the
      reference's in the same window, in at least `min_quarters` windows.

    Parameters
    ----------
    cells, quarters : pd.DataFrame
        The two tables, for the cells to judge (e.g. the zero-shot models only).
    alpha : float
        The one-sided significance level, e.g. 0.05.
    rule : CentreScaleRule
        `min_quarters` (e.g. 3 of the 4 test quarters), `main_family_min_quarters` (3) and
        `coverage_tolerance` (e.g. 0.05).

    Returns
    -------
    pd.DataFrame
        `cells` plus ``family``, ``positive_ic_quarters``, ``calibrated_quarters``,
        ``better_scale_quarters``, ``adjusted_p`` (Holm in the family; NaN without a test)
        and the bools ``has_centre_skill``, ``has_worse_centre``, ``is_calibrated``,
        ``beats_reference_scale`` and ``family_has_centre_skill`` (the family's Q1 verdict:
        any of its cells has centre skill).
    """
    better = (quarters["coverage_error"] < quarters["reference_coverage_error"]) & (
        quarters["ks_distance"] < quarters["reference_ks_distance"]
    )
    flags = quarters[["model", *CELL]].assign(
        positive_ic_quarters=quarters["rank_ic"] > 0,
        calibrated_quarters=quarters["coverage_error"] <= rule.coverage_tolerance + FLOAT_SLACK,
        better_scale_quarters=better,
    )
    counts = flags.groupby(["model", *CELL], as_index=False).sum()
    verdict = cells.merge(counts, on=["model", *CELL], how="left", validate="one_to_one")
    quarter_counts = ["positive_ic_quarters", "calibrated_quarters", "better_scale_quarters"]
    verdict = verdict.fillna(dict.fromkeys(quarter_counts, 0))
    is_scored = verdict["status"] == SCORED
    families = np.where(verdict["full_quarters"] >= rule.main_family_min_quarters, MAIN, LATE)
    adjusted = _family_holm(verdict["p_value"].to_numpy(dtype=float), families)
    has_centre_skill = (
        (verdict["hit_difference"] > 0)
        & (adjusted < alpha)
        & (verdict["positive_ic_quarters"] >= rule.min_quarters)
    )
    is_skilled = is_scored & has_centre_skill
    family_verdict = is_skilled.groupby(families).transform("any")
    return verdict.assign(
        family=families,
        adjusted_p=adjusted,
        has_centre_skill=is_skilled,
        family_has_centre_skill=family_verdict,
        has_worse_centre=is_scored & (verdict["diff_high"] < 0),
        is_calibrated=is_scored & (verdict["calibrated_quarters"] >= rule.min_quarters),
        beats_reference_scale=is_scored & (verdict["better_scale_quarters"] >= rule.min_quarters),
    )
