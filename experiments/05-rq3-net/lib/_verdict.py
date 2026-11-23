"""The decision rule of rq3: which zero-shot cells earn money after costs."""

import numpy as np
import pandas as pd

from src.benchmark import SCORED
from src.stats import holm

MAIN = "main"
"""The ``family`` of a model with enough post-release quarters for the main Holm family."""

LATE = "late"
"""The ``family`` of a late-release model: its own Holm family and its own verdict."""


def net_verdict(
    cells: pd.DataFrame,
    *,
    alpha: float,
    main_min_quarters: int,
    min_blocks: float,
    min_turnover: float,
) -> pd.DataFrame:
    """Apply the decision rule of rq3 to the cells of `net_tables`.

    A model is in the ``main`` family if it has at least `main_min_quarters` scored folds
    that start on or after its release (``post_release_quarters``), else in the ``late``
    family. The one-sided p-values (H1: the mean net return is above 0) are Holm-adjusted
    within each family, over all its horizons and markets (a cell without a test has p NaN
    and does not count). A scored cell earns money after costs if its net Sharpe ratio is
    positive and its adjusted p-value is below `alpha`. Two flags, on scored cells only, do
    not change the verdict: a cell whose bootstrap draws hold fewer than `min_blocks` blocks
    (``has_few_blocks``) and a book whose turnover per effective time (its total turnover
    over ``effective_times``, so the same scale at every horizon) is below `min_turnover`
    (``is_low_turnover``: its sign almost never changes, so it is close to buy and hold and
    its break-even cost is gross over almost no turnover).

    Parameters
    ----------
    cells : pd.DataFrame
        The ``"cells"`` table of `net_tables`, for the cells to judge (the zero-shot models).
    alpha : float
        The significance level, e.g. 0.05.
    main_min_quarters : int
        The fewest post-release quarters of a model in the main family, e.g. 3 of 4.
    min_blocks : float
        The fewest bootstrap blocks per draw before a cell is flagged.
    min_turnover : float
        The lowest turnover per effective time before a cell is flagged as low turnover.

    Returns
    -------
    pd.DataFrame
        `cells` plus ``family``, ``adjusted_p`` (Holm within the family),
        ``turnover_per_effective``, ``earns_after_costs``, ``has_few_blocks`` and
        ``is_low_turnover`` (bools).
    """
    family = np.where(cells["post_release_quarters"] >= main_min_quarters, MAIN, LATE)
    verdict = cells.assign(family=family)
    adjusted = pd.Series(np.nan, index=verdict.index)
    for _, group in verdict.groupby("family"):
        adjusted[group.index] = holm(group["p_value"].to_numpy(dtype=float))
    earns = (verdict["net_sharpe"] > 0) & (adjusted < alpha)
    per_effective = verdict["turnover"] * verdict["times"] / verdict["effective_times"]
    return verdict.assign(
        adjusted_p=adjusted,
        turnover_per_effective=per_effective,
        earns_after_costs=earns,
        has_few_blocks=(verdict["status"] == SCORED) & (verdict["blocks"] < min_blocks),
        is_low_turnover=(verdict["status"] == SCORED) & (per_effective < min_turnover),
    )
