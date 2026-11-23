"""The table of the figure of rq2 (``fig:rq2``): the sign hit rate and the coverage per model.

`figure_table` takes the verdict rows at one horizon and gives ``tables/rq2.csv``, from which
`src.figures.draw_centre_scale` draws ``figures/rq2.pdf``.
"""

import pandas as pd

from src.benchmark import SCORED
from src.figures import MARKET_ORDER

from ._centre_scale import TARGET_COVERAGE, TESTED

MARKET_LABELS = dict(
    zip(["crypto", "forex", "commodity", "index", "stock"], MARKET_ORDER, strict=True)
)
"""The label of each market of the cache, in the paper's market order."""

FIGURE_COLUMNS = [
    "model",
    "kind",
    "release_date",
    "family",
    "market",
    "horizon",
    "status",
    "times",
    "signed_decisions",
    "signed_times",
    "hit_rate",
    "hit_low",
    "hit_high",
    "reference_hit_rate",
    "has_centre_skill",
    "centre_test",
    "is_centre_tested",
    "coverage_gap",
    "coverage_gap_low",
    "coverage_gap_high",
    "is_calibrated",
]
"""The columns of ``tables/rq2.csv``."""


def figure_table(verdict: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    """Return the rows of the figure of rq2: one per model and market at `horizon`.

    Baselines come first, then the zero-shot models in release order; the markets in the
    paper's order. ``coverage_gap`` is the coverage minus 0.80, with its interval. A cell
    that is not scored keeps its row and ``status`` with NaN marks, so it is not drawn.

    Parameters
    ----------
    verdict : pd.DataFrame
        Rows of `centre_scale_verdict` (the zero-shot cells and the scale reference's).
    horizon : int
        The horizon in bars.

    Returns
    -------
    pd.DataFrame
        ``FIGURE_COLUMNS``; ``market`` is the paper's label of the market.

    Raises
    ------
    ValueError
        If no row has `horizon`, or a market has no label in `MARKET_LABELS`.
    """
    rows = verdict[verdict["horizon"] == horizon]
    if rows.empty:
        raise ValueError(
            f"no cell at horizon {horizon}; horizons {sorted(set(verdict['horizon']))}"
        )
    unknown = sorted(set(rows["market"]) - set(MARKET_LABELS))
    if unknown:
        raise ValueError(f"markets {unknown} have no label; expected {sorted(MARKET_LABELS)}")
    # a cell not scored keeps its row and status but has no mark, so it never looks like a
    # scored cell that failed the rule
    plotted = ["hit_rate", "reference_hit_rate", "coverage"]
    rows = rows.assign(**{
        column: rows[column].where(rows["status"] == SCORED) for column in plotted
    })  # fmt: skip
    gaps = {
        "coverage_gap": rows["coverage"] - TARGET_COVERAGE,
        "coverage_gap_low": rows["coverage_low"] - TARGET_COVERAGE,
        "coverage_gap_high": rows["coverage_high"] - TARGET_COVERAGE,
    }
    table = rows.assign(
        market=rows["market"].map(MARKET_LABELS),
        is_centre_tested=rows["centre_test"] == TESTED,
        is_baseline=rows["kind"] == "baseline",
        market_order=rows["market"].map(list(MARKET_LABELS).index),
        **gaps,
    )
    order = ["is_baseline", "release_date", "model", "market_order"]
    table = table.sort_values(order, ascending=[False, True, True, True], na_position="first")
    return table[FIGURE_COLUMNS].reset_index(drop=True)
