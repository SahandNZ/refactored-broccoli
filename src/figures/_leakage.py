"""The figure of RQ4 (``fig:rq4``): the difference in differences of every model and market.

Drawn from ``tables/rq4.csv`` of 06-rq4-leakage (its ``figure_table``) by that study's
analysis.
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from src.figures._style import (
    DOUBLE_WIDTH,
    INTERVAL_COLOUR,
    MARK_COLOUR,
    MARKET_ORDER,
    REFERENCE_LINE,
    plain_tick,
    save_figure,
    shade_rows,
    use_paper_style,
)

BASE_HEIGHT = 0.75
"""Height in inches of the figure around its rows: the panel titles, the x-axis ticks and the
shared x label."""

ROW_HEIGHT = 0.2
"""Height in inches of one model's row."""

CUT_MARKERS = {-1: "<", 1: ">"}
"""The arrowhead drawn at the axis edge where an interval runs past `x_limit`."""


def draw_leakage(
    table: pd.DataFrame,
    *,
    labels: dict[str, str],
    market_labels: dict[str, str],
    x_limit: float,
    pdf_path: Path,
    grey_path: Path,
) -> None:
    """Draw the figure of RQ4: the difference in differences of the CRPS gain per model and market.

    Small multiples (charts.md 2a.4): one panel per market of `MARKET_ORDER` that the table has,
    left to right, titled with its label, sharing both axes; one row per model, top to bottom in
    the order of `labels` (the caller's: tab:rq1's order) and labelled with its values (the
    paper names), every second row on a light band (`shade_rows`); so each row of a panel holds
    one mark. x is the difference in differences of the CRPS gain over the selected baseline
    (baseline CRPS minus model CRPS, in the units of the target), before the release minus after
    it: positive is what leakage predicts. Marks are near-black circles (rows already name the
    models, charts.md 2a.3), filled when the cell shows leakage (the rule of rq4, Holm-adjusted);
    lines: 95 % intervals, unadjusted, in a lighter grey. A NaN mark (a cell not scored) is not
    drawn. The x-axis runs from -`x_limit` to `x_limit`; an interval past it stops at the edge
    with an arrowhead, a mark past it is not drawn. A dashed line at 0. No legend: the caption
    names the encodings. The figure is 6.80 in wide and 0.75 + 0.2 in per model high, in the
    paper style.

    Parameters
    ----------
    table : pd.DataFrame
        One row per model and market: ``model``, ``market`` (a label of `MARKET_ORDER`),
        ``horizon`` (one value, named in the axis label), ``did``, ``low``, ``high`` and
        ``shows_leakage`` (06-rq4-leakage ``tables/rq4.csv``).
    labels : dict of str to str
        Every model of `table` with its label, in the order of the rows, top to bottom.
    market_labels : dict of str to str
        The panel title of each market of `MARKET_ORDER` (the paper's market names).
    x_limit : float
        The half-width of the x-axis, positive.
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.

    Raises
    ------
    KeyError
        If a market of `table` is not in `MARKET_ORDER` or has no label.
    ValueError
        If a model of `table` has no label, the table has more than one horizon, or `x_limit`
        is not positive.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    present = set(table["model"].astype(str))
    missing = sorted(present - set(labels))
    if missing:
        raise ValueError(f"models {missing} of the table have no label")
    horizons = sorted(set(table["horizon"].astype(int)))
    if len(horizons) != 1:
        raise ValueError(f"the table must have one horizon, got {horizons}")
    horizon = horizons[0]
    if not x_limit > 0:
        raise ValueError(f"x_limit must be positive, got {x_limit}")
    unknown = sorted(set(table["market"].astype(str)) - set(MARKET_ORDER))
    if unknown:
        raise KeyError(f"markets {unknown} are not in MARKET_ORDER")
    markets = [market for market in MARKET_ORDER if market in set(table["market"].astype(str))]
    titles = {market: market_labels[market] for market in markets}
    models = [name for name in labels if name in present]
    height = BASE_HEIGHT + ROW_HEIGHT * len(models)
    # the paper style is set only inside this block, so the caller's settings stay unchanged
    with mpl.rc_context():
        use_paper_style()
        figure, axes = plt.subplots(
            1, len(markets), figsize=(DOUBLE_WIDTH, height), sharex=True, sharey=True,
            squeeze=False,
        )  # fmt: skip
        for axis, market in zip(axes[0], markets, strict=True):
            shade_rows(axis, len(models))
            cells = table[table["market"].astype(str) == market]
            _draw_cells(axis, cells, models, x_limit)
            axis.axvline(0.0, **REFERENCE_LINE)
            axis.set_title(titles[market])
            axis.tick_params(axis="y", length=0)
        first = axes[0][0]
        first.set_xlim(-x_limit, x_limit)
        # three ticks (-a, 0, a) without trailing zeros, so narrow panels keep their labels apart
        first.xaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True, steps=[1, 2, 5, 10]))
        first.xaxis.set_major_formatter(FuncFormatter(plain_tick))
        first.set_yticks(range(len(models)), [labels[name] for name in models])
        first.set_ylim(len(models) - 0.5, -0.5)
        bars = "bar" if horizon == 1 else "bars"
        figure.supxlabel(
            "Difference in differences of the CRPS gain over the selected baseline, "
            f"{horizon} {bars}: before minus after the release (> 0: larger pre-release gain)"
        )
        save_figure(figure, pdf_path, grey_path)


def _draw_cells(axis: "Axes", cells: pd.DataFrame, models: list[str], x_limit: float) -> None:
    # one market's marks and intervals, one per model row; an interval past the edge stops there
    # with an arrowhead, a mark past it is not drawn, a cell not scored has no mark
    for model, did, low, high, is_leaky in zip(
        cells["model"].astype(str),
        cells["did"].to_numpy(dtype=float),
        cells["low"].to_numpy(dtype=float),
        cells["high"].to_numpy(dtype=float),
        cells["shows_leakage"].to_numpy(dtype=bool),
        strict=True,
    ):
        if np.isnan(did):
            continue
        y = models.index(model)
        axis.plot(
            [max(low, -x_limit), min(high, x_limit)], [y, y],
            color=INTERVAL_COLOUR, linewidth=1.0, solid_capstyle="butt",
        )  # fmt: skip
        for side, bound in ((-1, low), (1, high)):
            if side * bound > x_limit:
                axis.plot(
                    side * x_limit, y, CUT_MARKERS[side], color=INTERVAL_COLOUR,
                    markersize=3.5, clip_on=False,
                )  # fmt: skip
        if abs(did) <= x_limit:
            fill = MARK_COLOUR if is_leaky else "white"
            axis.plot(
                did, y, "o", color=MARK_COLOUR, markerfacecolor=fill, markersize=4,
                markeredgewidth=1.0, zorder=3,
            )  # fmt: skip
