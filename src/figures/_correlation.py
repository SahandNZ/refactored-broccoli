"""Figure of the asset universe: the correlation matrix of the assets, grouped by market.

Drawn from ``tables/universe_correlation.csv`` by the analysis of 01-universe and by
``paper/scripts/make_fig_correlation.py``, in the owner's design (2026-11-19), drawn at the
paper's column width so that its text keeps its size in the paper (the owner's final pass).
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.figures._style import check_figure, plain_tick, save_figure, serif_rc

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

FIGURE_SIZE = (3.46, 2.37)
"""Width and height in inches of the drawn figure: cropped to its content it is as wide as the
paper's column (239.4 pt), so it is included unscaled."""

MATRIX_BOX = (0.18, 0.06, 0.64, 0.88)
"""Left, bottom, width and height of the matrix axes, as fractions of the figure."""

BAR_BOX = (1.035, 0.0, 0.041, 1.0)
"""The colour bar right of the matrix and as tall as it, in fractions of the matrix axes."""

LABEL_FONT_SIZE = 7.0
"""Size in points of the market names and the colour bar ticks, in the paper: below the
AAAI-27 9 pt limit, the size the owner chose for this appendix figure (charts.md 2a.14)."""

BAR_LABEL_SIZE = 7.5
"""Size in points of the colour bar's label."""

INK = "#172638"
"""The dark blue of every text and tick."""

SEPARATOR_COLOUR = "#3C4654"
"""The grey of the lines that part the markets of the matrix (drawn at `SEPARATOR_ALPHA`)."""

SEPARATOR_ALPHA = 0.62
"""The opacity of the market lines."""

CROP_PAD = 0.01
"""Padding in inches around the drawn content when the figure is cropped."""

PNG_DPI = 360
"""Resolution of the PNG copy."""

STYLE: "dict[RcKeyType, str | int]" = {
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
}
"""The matplotlib settings of this figure, on top of matplotlib's defaults and the serif."""


def _market_blocks(markets: list[str]) -> list[tuple[str, int, int]]:
    # (market, first row, last row) of every market, in the order the markets first appear
    blocks: list[tuple[str, int, int]] = []
    for market in markets:
        if blocks and blocks[-1][0] == market:
            continue
        rows = [row for row, name in enumerate(markets) if name == market]
        if rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError(f"the assets of market {market!r} must be next to each other")
        blocks.append((market, rows[0], rows[-1]))
    return blocks


def _check_matrix(matrix: pd.DataFrame) -> np.ndarray:
    # the values of a finite, symmetric correlation matrix with a unit diagonal
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("matrix must have the same assets as index and columns, in one order")
    values = matrix.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.max(np.abs(values)) > 1 + 1e-10
        or not np.allclose(values, values.T, atol=1e-10, rtol=0)
        or not np.allclose(np.diag(values), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("expected a finite, symmetric correlation matrix with a unit diagonal")
    return values


def draw_correlation(
    matrix: pd.DataFrame,
    markets: list[str],
    *,
    pdf_path: Path,
    grey_path: Path,
    png_path: Path | None = None,
) -> None:
    """Draw a correlation matrix as a heatmap, with the assets grouped and labelled by market.

    The owner's design (2026-11-19): one cell per pair of assets, the whole matrix, a grey line
    across it where each market ends, the market names left of it and no column names; the
    colour map is diverging and centred on 0 (red positive, blue negative), its bar upright
    right of the matrix and as tall as it. The figure is drawn at `FIGURE_SIZE` in TeX Gyre
    Termes and cropped to its content, as wide as the paper's column.

    Parameters
    ----------
    matrix : pd.DataFrame
        The correlations, shape (n_assets, n_assets), with the same assets as index and
        columns, in the order to draw.
    markets : list of str
        The market of each asset, in the order of `matrix`; each market's assets are next to
        each other.
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.
    png_path : Path or None
        A colour PNG copy to write at `PNG_DPI`; None writes none.

    Raises
    ------
    ValueError
        If `matrix` is not square with equal index and columns, is not a finite symmetric
        correlation matrix with a unit diagonal, `markets` has a different length, or the
        assets of a market are not next to each other.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import Bbox

    values = _check_matrix(matrix)
    if len(markets) != len(matrix):
        raise ValueError(f"markets has {len(markets)} entries, expected {len(matrix)}")
    blocks = _market_blocks(markets)
    size = len(values)
    # the style is set only inside this block, on matplotlib's defaults, so the caller's
    # settings neither change nor leak into the figure
    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update(serif_rc())
        mpl.rcParams.update(STYLE)
        figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
        axis = figure.add_axes(MATRIX_BOX)
        image = axis.imshow(
            values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
            extent=(0, size, size, 0),
        )  # fmt: skip
        for _, _, last in blocks[:-1]:
            axis.axhline(last + 1, color=SEPARATOR_COLOUR, alpha=SEPARATOR_ALPHA, lw=0.55)
            axis.axvline(last + 1, color=SEPARATOR_COLOUR, alpha=SEPARATOR_ALPHA, lw=0.55)
        centres = [(first + last + 1) / 2 for _, first, last in blocks]
        axis.set_xticks([])
        axis.set_yticks(centres, [name for name, _, _ in blocks], fontsize=LABEL_FONT_SIZE)
        axis.tick_params(axis="y", length=0, pad=4)
        for spine in axis.spines.values():
            spine.set_visible(False)
        bar_axis = axis.inset_axes(BAR_BOX)
        bar = figure.colorbar(
            image, cax=bar_axis, orientation="vertical", ticks=[-1, -0.5, 0, 0.5, 1],
            format=FuncFormatter(plain_tick),
        )  # fmt: skip
        bar.set_label("Correlation of weekly log returns", fontsize=BAR_LABEL_SIZE, labelpad=4)
        bar.ax.tick_params(length=2.5, width=0.5, labelsize=LABEL_FONT_SIZE, pad=2)
        bar.outline.set_linewidth(0.5)
        # crop to the content with a thin margin, so LaTeX sets the space around the figure
        figure.draw_without_rendering()
        content = figure.get_tightbbox()
        crop = Bbox.from_extents(
            content.x0 - CROP_PAD, content.y0 - CROP_PAD,
            content.x1 + CROP_PAD, content.y1 + CROP_PAD,
        )  # fmt: skip
        check_figure(figure, min_font_size=LABEL_FONT_SIZE)
        if png_path is not None:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(png_path, dpi=PNG_DPI, bbox_inches=crop)
        save_figure(figure, pdf_path, grey_path, min_font_size=LABEL_FONT_SIZE, crop=crop)
