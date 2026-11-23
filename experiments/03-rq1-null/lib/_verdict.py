import numpy as np
import pandas as pd

from src.benchmark import beats_baseline
from src.stats import holm


def null_verdict(cells: pd.DataFrame, *, alpha: float, min_winning_quarters: int) -> pd.DataFrame:
    """Apply the decision rule of rq1 to the cells of `src.benchmark.power_tables`.

    A scored cell beats the best baseline if its mean CRPS is lower in at least
    `min_winning_quarters` windows and its one-sided Diebold-Mariano test favours the model
    with a Holm-adjusted p-value below `alpha`. Holm runs within each family (``family``:
    ``main``, the models with enough quarters after their scoring start, and ``late``), over
    the scored cells of that family only (`src.stats.holm` leaves the cells without a test
    out of ``m``).

    Parameters
    ----------
    cells : pd.DataFrame
        The ``"cells"`` table of `power_tables`, one row per model, market and horizon.
    alpha : float
        The significance level, e.g. 0.05.
    min_winning_quarters : int
        The fewest windows the model must win, e.g. 3 of the 4 test quarters.

    Returns
    -------
    pd.DataFrame
        `cells` plus ``adjusted_p`` (Holm within the family; NaN for a cell not scored) and
        ``beats_baseline``.
    """
    adjusted = np.full(len(cells), np.nan)
    for _, rows in cells.groupby("family").indices.items():
        adjusted[rows] = holm(cells["p_value"].to_numpy(dtype=float)[rows])
    beats = beats_baseline(
        adjusted,
        cells["winning_quarters"].to_numpy(),
        cells["dm_statistic"].to_numpy(),
        alpha=alpha,
        min_winning_quarters=min_winning_quarters,
    )
    return cells.assign(adjusted_p=adjusted, beats_baseline=beats)
