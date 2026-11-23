"""The tests and the decision rules of 08-closures (Q1 and Q2).

`closure_tests` compares, for each model on the closed markets, forward-filled and missing with
the reference, trading hours only, on the series of closure days; `closure_verdict` applies
Q1's rule in two Holm families (models on time and late models); `best_representations` sets
each model's best representation on the closed markets next to its best on the control (Q2).
"""

import math
import zlib

import numpy as np
import pandas as pd

from src.benchmark import SCORED
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    block_indices,
    diebold_mariano,
    has_enough_blocks,
    holm,
    politis_white_block,
)

from ._closure_contexts import FORWARD_FILLED, MISSING, REPRESENTATIONS, TRADING_HOURS
from ._closures import ALL, BASELINE, CLOSED, CONTROL, ClosureSettings, closure_day_means

REFERENCE = TRADING_HOURS
"""The reference representation of the tests, fixed before any run."""

ALTERNATIVES = (FORWARD_FILLED, MISSING)
"""The representations tested against the reference."""

ON_TIME = "on time"
LATE = "late"
"""The two Holm families: models with at least ``min_quarters`` of the scored quarters
starting at or after their scoring start (`src.benchmark.full_quarters`), and the others."""

OVERLAP = 0
"""Label overlap of the closure-day series: every label is one step ahead, and the target
bars of one closure count on one day, so no two days share a label."""


def block_length(difference: np.ndarray, *, overlap: int) -> int:
    """Return the moving-block length of a series: Politis-White, floored by its overlap.

    `src.stats.politis_white_block` (1 for a series without variance or too short to
    estimate), at least ``max(1, BLOCK_OVERLAP_FLOOR * overlap)`` (`src.stats`; with the
    closure-day series' overlap of 0 the overlap floor is 0, so only the minimum block of 1
    acts), at most the series' length.
    A series with fewer than `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks is not tested
    (`closure_tests`).

    Parameters
    ----------
    difference : np.ndarray
        The series in time order, shape ``(n,)``, n >= 1.
    overlap : int
        How many later values share a label with each value.

    Returns
    -------
    int
        The block length in values.
    """
    floor = max(1, BLOCK_OVERLAP_FLOOR * overlap)
    return min(max(floor, politis_white_block(difference)), len(difference))


def skill_interval(
    alternative: np.ndarray,
    reference: np.ndarray,
    *,
    block: int,
    level: float,
    n_resamples: int,
    seed: int | list[int],
) -> tuple[float, float]:
    """Return the moving-block bootstrap interval of the skill ``1 - mean(alt) / mean(ref)``.

    The same blocks of days resample both series, so the interval keeps their pairing and
    their autocorrelation within a block.

    Parameters
    ----------
    alternative, reference : np.ndarray
        Mean CRPS per closure day of the two representations, same days, shape ``(n,)``.
    block : int
        Block length in days (`block_length`).
    level : float
        Coverage, e.g. 0.95.
    n_resamples : int
        Bootstrap draws.
    seed : int or list of int
        Seed of the draws (`closure_tests` passes the study's seed with the model's checksum
        and the alternative, so every test has its own stream).

    Returns
    -------
    tuple of (float, float)
        The percentile interval of the skill (positive: the alternative is better).
    """
    rng = np.random.default_rng(seed)
    rows = block_indices(len(reference), rng, n_resamples=n_resamples, block=block)
    draws = 1 - alternative[rows].mean(axis=1) / reference[rows].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


TEST_COLUMNS = [
    "model",
    "family",
    "quarters",
    "alternative",
    "days",
    "reference_crps",
    "alternative_crps",
    "skill",
    "skill_low",
    "skill_high",
    "block",
    "dm_lags",
    "dm_statistic",
    "p_value",
    "status",
]
"""The columns of `closure_tests`."""


def _closure_days(frame: pd.DataFrame, model: str, column: str = "crps") -> dict[str, np.ndarray]:
    # each representation's mean `column` per closure day, on the same days in the same order
    series = {
        str(arm): closure_day_means(arm_frame, [column])[column]
        for arm, arm_frame in frame.groupby("representation")
    }
    days = series[REFERENCE].index
    for arm, values in series.items():
        if not values.index.equals(days):
            raise ValueError(f"{model}: {arm} is scored on other closure days than {REFERENCE}")
    return {arm: values.to_numpy() for arm, values in series.items()}


def _test(
    alternative: np.ndarray,
    reference: np.ndarray,
    settings: ClosureSettings,
    *,
    stream: list[int],
) -> dict[str, object]:
    # one alternative against the reference on the closure days: the skill, and with enough
    # days its interval and the one-sided Diebold-Mariano test (H1: the alternative is better)
    row: dict[str, object] = {
        "days": len(reference),
        "reference_crps": float(reference.mean()),
        "alternative_crps": float(alternative.mean()),
        "skill": float(1 - alternative.mean() / reference.mean()),
    }
    if len(reference) < settings.min_closure_days:
        return row | {"status": f"not tested: fewer than {settings.min_closure_days} closure days"}
    test = diebold_mariano(alternative, reference, overlap=OVERLAP)
    if math.isnan(test["statistic"]):
        return row | {"status": "not tested: the loss difference has no variance"}
    block = block_length(alternative - reference, overlap=OVERLAP)
    if not has_enough_blocks(len(reference), block):
        return row | {
            "block": block,
            "status": f"not tested: fewer than {MIN_BOOTSTRAP_BLOCKS} bootstrap blocks",
        }
    low, high = skill_interval(
        alternative,
        reference,
        block=block,
        level=1 - settings.alpha,
        n_resamples=settings.n_resamples,
        seed=[settings.bootstrap_seed, *stream],
    )
    return row | {
        "skill_low": low,
        "skill_high": high,
        "block": block,
        "dm_lags": test["lags"],
        "dm_statistic": test["statistic"],
        "p_value": test["p_less"],
        "status": SCORED,
    }


def closure_tests(losses: pd.DataFrame, settings: ClosureSettings) -> pd.DataFrame:
    """Test forward-filled and missing against trading hours only, per zero-shot model (Q1).

    On the closed markets, each representation's scored losses become a series of closure
    days (`closure_day_means`). Each alternative is compared with the reference by the
    one-sided Diebold-Mariano test (`src.stats.diebold_mariano`: Newey-West variance with
    ``max(overlap, Andrews)`` lags, Harvey-Leybourne-Newbold correction, Student's t with
    ``n - 1`` degrees of freedom; H1: the alternative has the lower expected CRPS), and its
    skill ``1 - CRPS_alt / CRPS_ref`` gets a moving-block bootstrap interval with the
    Politis-White block of the difference (`block_length`, `skill_interval`). A test gets
    neither with fewer than ``min_closure_days`` closure days, fewer than
    `src.stats.MIN_BOOTSTRAP_BLOCKS` blocks, a difference without variance, or no scored
    decision (``status`` says which). A model with fewer than
    ``min_quarters`` scored quarters starting at or after its scoring start is in the late
    family. Each test's bootstrap stream is seeded by the study's seed, a checksum of the
    model's name (CRC-32) and the alternative's place in `ALTERNATIVES`, so it does not depend
    on which other models a run holds or on their order.

    Parameters
    ----------
    losses : pd.DataFrame
        The ``"losses"`` table of `closure_losses`.
    settings : ClosureSettings
        The study's settings.

    Returns
    -------
    pd.DataFrame
        One row per model and alternative (`TEST_COLUMNS`): ``family`` and ``quarters``,
        ``days``, both mean CRPS over the days, ``skill`` with ``skill_low`` and
        ``skill_high``, the ``block`` and ``dm_lags``, ``dm_statistic`` (negative when the
        alternative is better), the one-sided ``p_value`` (NaN when not tested) and
        ``status`` (`SCORED` or why not tested).
    """
    closed = losses.loc[
        (losses["group"] == CLOSED)
        & (losses["model"] != BASELINE)
        & losses["representation"].isin(REPRESENTATIONS)
    ]
    rows: list[dict[str, object]] = []
    for model, frame in closed.groupby("model"):
        quarters = settings.post_release_quarters(str(model))
        family = LATE if quarters < settings.min_quarters else ON_TIME
        alternatives = [name for name in ALTERNATIVES if name in set(frame["representation"])]
        head: dict[str, object] = {"model": model, "family": family, "quarters": quarters}
        scored = frame.loc[frame["is_scored"]]
        if scored.empty:
            status = "not tested: no scored decision"
            rows += [
                head | {"alternative": name, "days": 0, "status": status} for name in alternatives
            ]
            continue
        days = _closure_days(scored, str(model))
        for alternative in alternatives:
            stream = [zlib.crc32(str(model).encode()), ALTERNATIVES.index(alternative)]
            test = _test(days[alternative], days[REFERENCE], settings, stream=stream)
            rows.append(head | {"alternative": alternative} | test)
    return pd.DataFrame(rows, columns=TEST_COLUMNS)


def _model_verdict(tests: pd.DataFrame) -> str:
    # among the tested alternatives: the reference unless one is significantly better, of two
    # the larger skill; the reason when none of them is tested
    is_tested = tests["status"] == SCORED
    if not is_tested.any():
        return "; ".join(dict.fromkeys(tests["status"].astype(str)))
    better = tests.loc[is_tested & tests["is_better"]]
    if better.empty:
        return REFERENCE
    skills = better["skill"].to_numpy(dtype=float)
    return str(better["alternative"].to_numpy()[np.argmax(skills)])


def closure_verdict(tests: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    """Apply the rule of Q1: keep trading hours only unless an alternative is better.

    The one-sided p-values are Holm-adjusted within each family (`ON_TIME`, `LATE`) over
    every test of its models (`src.stats.holm`). An alternative is better when its adjusted
    p-value is below `alpha`; a model's verdict is the better alternative (of two, the one
    with the larger skill) or else the reference. An alternative is worse when its whole
    skill interval is below zero (descriptive, not tested).

    Parameters
    ----------
    tests : pd.DataFrame
        The table of `closure_tests`.
    alpha : float
        The one-sided significance level.

    Returns
    -------
    pd.DataFrame
        `tests` plus ``adjusted_p``, ``is_better``, ``is_worse`` and ``verdict`` (the same on
        every row of a model: the representation to feed, decided by its tested alternatives,
        or why none is tested).
    """
    adjusted = np.full(len(tests), np.nan)
    p_values = tests["p_value"].to_numpy(dtype=float)
    for rows in tests.groupby("family").indices.values():
        adjusted[rows] = holm(p_values[rows])
    table = tests.assign(
        adjusted_p=adjusted,
        is_better=adjusted < alpha,
        is_worse=tests["skill_high"].to_numpy(dtype=float) < 0,
    )
    verdicts = {str(model): _model_verdict(rows) for model, rows in table.groupby("model")}
    return table.assign(verdict=table["model"].map(verdicts))


ROBUSTNESS_COLUMNS = [
    "model",
    "family",
    "alternative",
    "decisions",
    "days",
    "skill",
    "decisions_sorted",
    "days_sorted",
    "skill_sorted",
]
"""The columns of `sorted_robustness`."""


def sorted_robustness(losses: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Set each tested skill next to the same skill with sorted quantiles (descriptive).

    A model may give quantiles that cross; the primary rule counts such a forecast as invalid
    and drops its decision from every representation. Sorting the nine quantiles (the monotone
    rearrangement of Chernozhukov, Fernández-Val and Galichon, 2010) keeps a crossed but
    finite forecast. On the closed markets, each zero-shot model's skill
    ``1 - CRPS_alt / CRPS_ref`` over closure days is recomputed on the sorted forecasts and
    their own scored decisions (``crps_sorted``, ``is_scored_sorted``). No test, no interval,
    no Holm, no bold.

    Parameters
    ----------
    losses : pd.DataFrame
        The ``"losses"`` table of `closure_losses`.
    tests : pd.DataFrame
        The table of `closure_tests` (its ``skill``, ``days`` and ``family``).

    Returns
    -------
    pd.DataFrame
        One row per model and alternative of `tests` (`ROBUSTNESS_COLUMNS`): the scored
        decisions, closure days and skill of the primary rule and of the sorted quantiles.
    """
    closed = losses.loc[
        (losses["group"] == CLOSED)
        & (losses["model"] != BASELINE)
        & losses["representation"].isin(REPRESENTATIONS)
    ]
    rows: list[dict[str, object]] = []
    for model, frame in closed.groupby("model"):
        primary = frame.loc[frame["is_scored"] & (frame["representation"] == REFERENCE)]
        scored = frame.loc[frame["is_scored_sorted"]]
        days = _closure_days(scored, str(model), "crps_sorted") if len(scored) else {}
        reference_sorted = scored.loc[scored["representation"] == REFERENCE]
        for _, test in tests.loc[tests["model"] == model].iterrows():
            alternative = str(test["alternative"])
            skill = np.nan
            if alternative in days:
                skill = float(1 - days[alternative].mean() / days[REFERENCE].mean())
            rows.append(
                {"model": model, "family": test["family"], "alternative": alternative,
                 "decisions": len(primary), "days": test["days"], "skill": test["skill"],
                 "decisions_sorted": len(reference_sorted),
                 "days_sorted": len(days.get(REFERENCE, [])), "skill_sorted": skill}
            )  # fmt: skip
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)


def best_representations(summary: pd.DataFrame) -> pd.DataFrame:
    """Set each model's best representation on the closed markets next to its best on the control.

    The best representation of a model in a group is its scored one with the lowest mean CRPS
    over closure days (split `ALL` of the summary; Q2: the same best in both groups is
    consistent with an effect of the representation rather than of the market). The
    benchmark input is not a representation of the same hours and is left out.

    Parameters
    ----------
    summary : pd.DataFrame
        The ``"summary"`` table of `closure_losses`.

    Returns
    -------
    pd.DataFrame
        One row per zero-shot model scored in both groups: ``model``, ``closed_best``,
        ``control_best`` and ``is_same``.
    """
    scored = summary.loc[
        (summary["status"] == SCORED)
        & (summary["split"] == ALL)
        & (summary["model"] != BASELINE)
        & summary["representation"].isin(REPRESENTATIONS)
    ]
    best = scored.loc[scored.groupby(["model", "group"])["crps"].idxmin()]
    table = best.pivot(index="model", columns="group", values="representation")
    table = table.reindex(columns=[CLOSED, CONTROL]).dropna()
    table = table.rename(columns={CLOSED: "closed_best", CONTROL: "control_best"})
    table["is_same"] = table["closed_best"] == table["control_best"]
    return table.reset_index()[["model", "closed_best", "control_best", "is_same"]]


def family_answer(verdict: pd.DataFrame, family: str) -> str:
    """Answer Q1 for one Holm family: does any of its tested models feed better otherwise.

    Parameters
    ----------
    verdict : pd.DataFrame
        The table of `closure_verdict`.
    family : str
        `ON_TIME` (the answer of Q1) or `LATE` (its own verdict, reported next to it).

    Returns
    -------
    str
        ``"yes"`` when at least one model with a tested alternative has the verdict of an
        alternative, ``"no"`` when every such model keeps `REFERENCE`, ``"inconclusive"``
        when no model of the family has a tested alternative.
    """
    rows = verdict.loc[(verdict["family"] == family) & (verdict["status"] == SCORED)]
    if rows.empty:
        return "inconclusive"
    return "yes" if (rows["verdict"] != REFERENCE).any() else "no"


def control_answer(best: pd.DataFrame) -> str:
    """Answer Q2: ``"yes"`` when every model has the same best on both groups.

    Parameters
    ----------
    best : pd.DataFrame
        The table of `best_representations`.

    Returns
    -------
    str
        ``"yes"``, ``"no"``, or ``"inconclusive"`` without a model scored in both groups.
    """
    if best.empty:
        return "inconclusive"
    return "yes" if best["is_same"].all() else "no"
