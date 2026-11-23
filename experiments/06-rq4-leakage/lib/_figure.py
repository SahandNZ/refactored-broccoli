"""The table of the figure of rq4 (``fig:rq4``): the difference in differences per model.

`figure_table` takes the verdict rows of the test ``leakage`` at one horizon and gives
``tables/rq4.csv``, from which `src.figures.draw_leakage` draws ``figures/rq4.pdf``;
`figure_families` gives each model the Holm family of rq1 and rq2, so the rows have the order
of tab:rq1 and fig:rq2 (`src.benchmark.paper_order`).
"""

import pandas as pd

from src.benchmark import SCORED, full_quarters, paper_order
from src.evaluate import Fold
from src.figures import MARKET_ORDER

MARKET_LABELS = dict(
    zip(["crypto", "forex", "commodity", "index", "stock"], MARKET_ORDER, strict=True)
)
"""The key of each market of the cache in the figure code (`MARKET_ORDER`, which sets the
markers and the order); the paper's display names (`src.benchmark.MARKET_NAMES`) are the
legend's, passed by ``analyze.py``."""

WIDE_FACTOR = 3.0
"""A model whose widest interval is more than this many times the median model's is cut at
the edge of fig:rq4 (`figure_limit`), so it does not squeeze the other models' intervals."""

LIMIT_MARGIN = 1.08
"""The x-axis of fig:rq4 runs this factor past its widest kept value."""

FIGURE_COLUMNS = [
    "model",
    "row_group",
    "market",
    "horizon",
    "status",
    "first_times",
    "second_times",
    "first_block",
    "second_block",
    "did",
    "low",
    "high",
    "adjusted_p",
    "shows_leakage",
]
"""The columns of ``tables/rq4.csv``."""


def figure_families(
    models: pd.DataFrame, folds: tuple[Fold, ...], *, min_quarters: int
) -> dict[str, str]:
    """Return the Holm family of rq1 and rq2 of each model: the row order of their figures.

    rq4 tests every model in one family; the family here only orders the rows as in tab:rq1
    and fig:rq2: ``main`` when at least `min_quarters` of `folds` start at or after the
    model's release (`src.benchmark.full_quarters`), else ``late``.

    Parameters
    ----------
    models : pd.DataFrame
        ``models.csv`` of the forecast cache: ``model`` and ``release_date`` (empty for a
        baseline, which counts as ``main``).
    folds : tuple of Fold
        The scored folds.
    min_quarters : int
        rq1's ``main_family_min_quarters``.

    Returns
    -------
    dict of str to str
        Model name to ``main`` or ``late``.
    """
    releases = pd.to_datetime(models["release_date"], utc=True)
    return {
        str(name): "main" if pd.isna(release) or full_quarters(release, folds) >= min_quarters
        else "late"
        for name, release in zip(models["model"], releases, strict=True)
    }  # fmt: skip


def figure_table(verdict: pd.DataFrame, families: dict[str, str], *, horizon: int) -> pd.DataFrame:
    """Return the rows of the figure of rq4: one per model and market at `horizon`.

    The rows of the test ``leakage``, models in the order of tab:rq1 and fig:rq2
    (`src.benchmark.paper_order` of `families`), markets in the paper's order. A cell that
    is not scored keeps its row and ``status`` with NaN marks, so it is not drawn.

    Parameters
    ----------
    verdict : pd.DataFrame
        Rows of `leakage_verdict`.
    families : dict of str to str
        The family of each model (`figure_families`).
    horizon : int
        The horizon in bars.

    Returns
    -------
    pd.DataFrame
        ``FIGURE_COLUMNS``; ``market`` is the cache's market key (``stock``, ...);
        ``row_group`` is the model's family of rq1 (``main`` or ``late``), which only orders
        the rows (rq4 tests every model in one Holm family).

    Raises
    ------
    ValueError
        If no row of the test ``leakage`` has `horizon`, a market has no label in
        `MARKET_LABELS`, or a model has no family.
    """
    rows = verdict[(verdict["test"] == "leakage") & (verdict["horizon"] == horizon)]
    if rows.empty:
        raise ValueError(
            f"no leakage cell at horizon {horizon}; horizons {sorted(set(verdict['horizon']))}"
        )
    unknown = sorted(set(rows["market"]) - set(MARKET_LABELS))
    if unknown:
        raise ValueError(f"markets {unknown} have no label; expected {sorted(MARKET_LABELS)}")
    no_family = sorted(set(rows["model"]) - set(families))
    if no_family:
        raise ValueError(f"models {no_family} have no family")
    # a cell not scored keeps its row and status but has no mark
    plotted = ["did", "low", "high"]
    rows = rows.assign(**{
        column: rows[column].where(rows["status"] == SCORED) for column in plotted
    })  # fmt: skip
    table = rows.assign(
        row_group=rows["model"].map(families),
        market_order=rows["market"].map(list(MARKET_LABELS).index),
    )
    # models in tab:rq1's order (the figure's rows, top to bottom), markets in the paper's
    order = paper_order({name: families[name] for name in table["model"].astype(str)})
    table = table.assign(model_order=table["model"].map(order.index))
    table = table.sort_values(["model_order", "market_order"], kind="stable")
    return table[FIGURE_COLUMNS].reset_index(drop=True)


def figure_limit(table: pd.DataFrame) -> float:
    """Return the half-width of the x-axis of fig:rq4.

    It covers every drawn mark (``did``) and every interval of the models that are not wide: a
    model is wide when its widest interval (``high - low``) is more than `WIDE_FACTOR` times the
    median of the models' widest intervals (Kronos at 1 bar, whose intervals are about twelve
    times the others': 0.13 against a median of 0.011 in the test run); a wide model's interval
    is cut at the edge. Times `LIMIT_MARGIN`.

    Parameters
    ----------
    table : pd.DataFrame
        Rows of `figure_table` (NaN marks of the cells not scored are ignored).

    Returns
    -------
    float
        The half-width, positive.

    Raises
    ------
    ValueError
        If no cell of `table` has a mark.
    """
    marked = table.dropna(subset=["did", "low", "high"])
    if marked.empty:
        raise ValueError("no cell of the figure has a mark")
    widest = (marked["high"] - marked["low"]).groupby(marked["model"]).max()
    narrow = widest.index[widest <= WIDE_FACTOR * widest.median()]
    kept = marked[marked["model"].isin(narrow)]
    reach = max(
        float(marked["did"].abs().max()),
        float(kept[["low", "high"]].abs().to_numpy().max()),
    )
    return LIMIT_MARGIN * reach if reach > 0 else 1.0
