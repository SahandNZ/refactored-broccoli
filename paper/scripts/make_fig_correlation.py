"""Draw the appendix correlation figure of the listed assets from the committed 01-universe results.

Reads ``tables/universe_correlation.csv`` (selection-window correlations of weekly log returns of
the 72 listed assets, in pick order within each pool) and ``tables/comparison_universes.csv``
(the pool of each listed asset, set ``Universe``) of the final 01-universe run, and draws them with
``src.figures.draw_correlation`` in the owner's design into ``figures/fig_correlation.pdf`` and
``figures/fig_correlation.png``.
Run from the lab root: ``.venv/bin/python paper/scripts/make_fig_correlation.py``.
"""

from pathlib import Path

import pandas as pd

from src.figures import draw_correlation

RUN = Path("experiments/01-universe/results/2026-11-17T0005-random-1000")
"""The final 01-universe run (commit d039ddb)."""

FIGURES = Path("paper/figures")
"""The folder of the paper's figures."""

POOL_NAMES = {
    "commodity": "Commodities",
    "crypto": "Crypto",
    "forex": "FX",
    "index": "Indices",
    "stock_non_us": "Non-US stocks",
    "stock_us": "US stocks",
}
"""The row label of each pool."""


def read_inputs(run: Path = RUN) -> tuple[pd.DataFrame, list[str]]:
    """Return the correlation matrix of the listed assets and the pool name of each row.

    Parameters
    ----------
    run : Path
        The 01-universe run folder.

    Returns
    -------
    tuple of pd.DataFrame and list of str
        The matrix, in the order of the CSV, and the label of each row's pool.

    Raises
    ------
    ValueError
        If the matrix and the listed set have different assets.
    """
    matrix = pd.read_csv(run / "tables" / "universe_correlation.csv", index_col=0)
    sets = pd.read_csv(run / "tables" / "comparison_universes.csv")
    pools = sets[sets["universe"] == "Universe"].set_index("symbol")["pool"]
    if set(pools.index) != set(matrix.index) or len(pools) != len(matrix):
        raise ValueError("the matrix and the listed set have different assets")
    return matrix, [POOL_NAMES[pools[symbol]] for symbol in matrix.index]


def main() -> None:
    """Draw the figure, with the rows labelled by pool."""
    matrix, labels = read_inputs()
    draw_correlation(
        matrix,
        labels,
        pdf_path=FIGURES / "fig_correlation.pdf",
        grey_path=Path("paper/scripts/out/fig_correlation_grey.png"),
        png_path=FIGURES / "fig_correlation.png",
    )


if __name__ == "__main__":
    main()
