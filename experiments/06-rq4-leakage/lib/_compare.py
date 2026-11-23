"""The tests of rq4: the difference of two periods' mean skill per cell, and the verdict.

`compare_periods` tests one cell (circular-block bootstrap standard error of each period,
with its own Politis-White block, and a t reference), `leakage_cells` runs one test over
every cell of the skills table, and `leakage_verdict` applies Holm per test family and the
decision rule of rq4.
"""

import math

import numpy as np
import pandas as pd

from src.benchmark import CELL, SCORED, Bootstrap, baseline_names
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BLOCK_VALUES,
    MIN_BOOTSTRAP_BLOCKS,
    bootstrap_difference,
    has_enough_blocks,
    holm,
    politis_white_block,
)

TESTS = {
    "leakage": ("before", "after"),
    "hourly": ("before", "after-hourly"),
    "cutoff": ("before", "between"),
    "placebo": ("early", "late"),
}
"""The tests of a cell: the difference of the mean skill of the first period minus the second.

``leakage`` answers Q1; ``hourly`` (the after period on every hourly decision), ``cutoff``
(models with a pretraining cutoff) and ``placebo`` (the placebo models) are reported checks,
each its own Holm family. ``placebo`` is two-sided (a drift either way breaks the assumption of
the difference in differences); the others are one-sided (H1: the first period is better).
"""

TWO_SIDED = ("placebo",)
"""The tests with a two-sided p-value."""

PERIODS = {
    "before": "before the release",
    "between": "between the cutoff and the release",
    "after": "after the release on the model's grid",
    "after-hourly": "after the release",
    "early": "before the placebo split",
    "late": "after the placebo split",
}
"""Every period of a test, with the words its ``status`` uses."""


def _period_block(values: np.ndarray, *, overlap: int, period: str) -> tuple[int, str]:
    # the block of one period of a test and its status: the Politis-White block, at least
    # BLOCK_OVERLAP_FLOOR label overlaps and 2, as in rq1; a period with no time, too few
    # times for the block estimate, or fewer than MIN_BOOTSTRAP_BLOCKS blocks leaves the
    # test unscored
    times = len(values)
    words = PERIODS[period]
    if times == 0:
        return 0, f"not scored: no decision {words}"
    if times < MIN_BLOCK_VALUES:
        return 0, f"not scored: {times} decision times {words}, fewer than {MIN_BLOCK_VALUES}"
    block = max(politis_white_block(values), BLOCK_OVERLAP_FLOOR * overlap, 2)
    if not has_enough_blocks(times, block):
        return block, (
            f"not scored: {times} decision times {words}, fewer than {MIN_BOOTSTRAP_BLOCKS} "
            f"bootstrap blocks of {block}"
        )
    return block, SCORED


def compare_periods(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    test: str,
    bootstrap: Bootstrap,
) -> dict[str, object]:
    """Return the difference of the mean skill of two periods of one cell, with its test.

    Each period's times are resampled on their own with circular blocks of their own length
    (`src.stats.bootstrap_difference`): the Politis-White block of the period's skills
    (`src.stats.politis_white_block`), at least ``src.stats.BLOCK_OVERLAP_FLOOR`` (4) times
    the period's label overlap and 2. The difference over its bootstrap standard error is
    read against Student's t with Welch's degrees of freedom from the two periods' block
    counts; the interval is the t interval at ``bootstrap.level``. The p-value is one-sided,
    H1: the first period's skill is higher (``p_greater``), and two-sided for the tests of
    `TWO_SIDED`. A period with no time, fewer than ``MIN_BLOCK_VALUES`` times or fewer than
    ``MIN_BOOTSTRAP_BLOCKS`` blocks (`src.stats.has_enough_blocks`) leaves the cell
    unscored.

    Parameters
    ----------
    first, second : pd.DataFrame
        The rows of the cell's ``"skills"`` table in the two periods of `test` (one per time,
        with ``skill`` and ``overlap``).
    test : str
        A test of `TESTS`.
    bootstrap : Bootstrap
        The study's bootstrap (draws, seed, level).

    Returns
    -------
    dict
        ``first_times``, ``second_times``, ``first_block``, ``second_block``,
        ``skill_first``, ``skill_second``, ``status``, ``did`` (first minus second), ``low``,
        ``high`` and ``p_value`` (NaN when not scored).
    """
    names = TESTS[test]
    result: dict[str, object] = {"status": SCORED}
    series: list[tuple[np.ndarray, int]] = []
    for side, name, rows in (("first", names[0], first), ("second", names[1], second)):
        values = rows.sort_values("time")["skill"].to_numpy(dtype=float)
        overlap = int(rows["overlap"].iloc[0]) if len(rows) > 0 else 0
        block, status = _period_block(values, overlap=overlap, period=name)
        result[f"{side}_times"], result[f"{side}_block"] = len(values), block
        result[f"skill_{side}"] = float(values.mean()) if len(values) > 0 else math.nan
        if result["status"] == SCORED:
            result["status"] = status
        series.append((values, block))
    result |= {"did": math.nan, "low": math.nan, "high": math.nan, "p_value": math.nan}
    if result["status"] != SCORED:
        return result
    tested = bootstrap_difference(
        series[0][0],
        series[1][0],
        level=bootstrap.level,
        n_resamples=bootstrap.n_resamples,
        seed=bootstrap.seed,
        blocks=(series[0][1], series[1][1]),
    )
    p_value = tested["p_value"] if test in TWO_SIDED else tested["p_greater"]
    return result | {
        "did": tested["difference"],
        "low": tested["low"],
        "high": tested["high"],
        "p_value": p_value,
    }


def leakage_cells(skills: pd.DataFrame, *, test: str, bootstrap: Bootstrap) -> pd.DataFrame:
    """One test of every model, market and horizon of `skills`: `compare_periods`.

    For ``test = "leakage"`` the difference is the difference in differences: the skill is
    already the model's CRPS gain over the best baseline, so a change that hits the model and
    the baseline alike cancels out, and a positive value means the model is better, beyond
    the baseline, on data it may have seen.

    Parameters
    ----------
    skills : pd.DataFrame
        The ``"skills"`` table of `leakage_tables`.
    test : str
        A test of `TESTS`.
    bootstrap : Bootstrap
        The study's bootstrap (draws, seed, level).

    Returns
    -------
    pd.DataFrame
        One row per model, market and horizon: ``model``, ``market``, ``horizon``,
        ``baseline`` and the columns of `compare_periods`; empty without skills.
    """
    first, second = TESTS[test]
    rows: list[dict[str, object]] = []
    for (model, market, horizon), cell in skills.groupby(["model", *CELL]):
        row: dict[str, object] = {"model": model, "market": market, "horizon": horizon}
        periods = cell["period"]
        row["baseline"] = baseline_names(cell.loc[periods.isin([first, second]), "baseline"])
        tested = compare_periods(
            cell[periods == first],
            cell[periods == second],
            test=test,
            bootstrap=bootstrap,
        )
        rows.append(row | tested)
    return pd.DataFrame(rows)


def leakage_verdict(cells: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    """Apply the decision rule of rq4 to the cells of `leakage_tables`, Holm per test family.

    The p-values of each test (``test``) are Holm-adjusted over that test's scored cells (a
    cell without a test has p NaN and does not count). A cell rejects when its adjusted
    p-value (one-sided; two-sided for the placebo) is below `alpha`; a cell of the test
    ``leakage`` that rejects shows leakage. The other tests are reported checks and never
    change the answer.

    Parameters
    ----------
    cells : pd.DataFrame
        The ``"cells"`` table of `leakage_tables`, with ``test`` and ``p_value``.
    alpha : float
        The significance level, e.g. 0.05.

    Returns
    -------
    pd.DataFrame
        `cells` plus ``adjusted_p``, ``rejects`` and ``shows_leakage`` (bool).
    """
    adjusted = pd.Series(np.nan, index=cells.index)
    for _, family in cells.groupby("test"):
        adjusted[family.index] = holm(family["p_value"].to_numpy(dtype=float))
    rejects = adjusted < alpha
    return cells.assign(
        adjusted_p=adjusted, rejects=rejects, shows_leakage=rejects & (cells["test"] == "leakage")
    )
