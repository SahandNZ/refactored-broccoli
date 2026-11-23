"""The figure of RQ2 (``fig:rq2``): the one-bar sign hit rate and coverage gap by market.

The owner's layout (``plot_rq2.py``, 2026-11-19), kept as he drew it: two blocks, (a) the
median's sign hit rate and (b) the 10-90 % coverage minus 80 %, each with one panel per market
and one row per model. It is drawn from ``tables/rq2.csv`` of 04-rq2-center-vs-scale (its
``figure_table``), read as text by `read_centre_scale`; it does not use the paper style of
`src.figures._style`.
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.figures._style import serif_rc

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CentreScaleRows = dict[tuple[str, str], dict[str, str]]
"""The rows of ``rq2.csv`` as text, by model and market."""

RQ2_MODELS: tuple[tuple[str, str], ...] = (
    ("ewma-normal", "EWMA-normal"),
    ("chronos-bolt-small", "Chronos-Bolt (S)"),
    ("chronos-bolt-base", "Chronos-Bolt (B)"),
    ("tirex-1.1", "TiRex"),
    ("moirai-2-small", "Moirai 2 (S)"),
    ("timesfm-2.5", "TimesFM 2.5"),
    ("chronos-2", "Chronos-2"),
    ("kronos-small", "Kronos (S)"),
    ("kronos-base", "Kronos (B)"),
    ("toto-2-22m", "Toto 2 (22M)"),
    ("toto-2-313m", "Toto 2 (313M)"),
    ("tirex-2", "TiRex-2"),
    ("timesfm-3", "TimesFM 3"),
    ("t0-gamma", "t0"),
)
"""The rows, top to bottom: the model's name in ``rq2.csv`` and its label (the order of tab:rq1:
the finance-pretrained Kronos checkpoints last in the main family)."""

RQ2_MARKETS: tuple[tuple[str, str], ...] = (
    ("Crypto", "o"),
    ("FX", "s"),
    ("Commodities", "^"),
    ("Indices", "D"),
    ("US stocks", "v"),
)
"""The panels, left to right: the market's label in ``rq2.csv`` and its marker."""

MARKET_TITLES: dict[str, str] = {"US stocks": "Stocks"}
"""The panel title of a market whose ``rq2.csv`` label is not its paper name: the stock market
holds US and non-US stocks (the label stays the key the rows are matched by)."""

BLUE = "#1669A5"
"""The colour of every mark and interval."""

INK = "#172638"
"""The colour of the text and of climatology's tick."""

GRID = "#E4E9EF"
"""A light grid colour (part of the owner's palette; not drawn)."""

REFERENCE = "#8593A3"
"""The colour of the reference line and of the "no estimate" dash."""

BAND = "#F3F6F9"
"""The colour of every second row's band."""

METRICS = (
    ("(a) Sign hit rate of the forecast median", "Sign hit rate (%)", "hit_rate", "hit_low",
     "hit_high", "is_centre_tested", (39.9, 60.6), (40, 50, 60), 50),
    ("(b) 10\u201390% interval coverage gap", "Coverage minus 80% (percentage points)",
     "coverage_gap", "coverage_gap_low", "coverage_gap_high", "is_calibrated",
     (-26.7, 10.8), (-20, -10, 0, 10), 0),
)  # fmt: skip
"""One block per metric: title, x label, value, low and high column, filled-mark column,
x range, x ticks and reference line (all in percent)."""

Metric = tuple[
    str, str, str, str, str, str, tuple[float, float], tuple[int, ...], int
]  # fmt: skip
"""One entry of `METRICS`."""


CROP_PAD = 0.01
"""Margin in inches left around the drawn content, so LaTeX sets the space around the figure."""

FIGURE_WIDTH = 7.0
"""Width of the figure in inches: the paper's ``\\textwidth`` (charts.md), so it is included at
natural size."""

LEFT = 1.25
"""Room for the model labels left of the first panel, in inches."""

RIGHT = 0.12
"""Room right of the last panel, in inches."""

GAP = 0.18
"""Space between two market panels, in inches."""

TITLE_X = 0.06
"""Left edge of a block title, in inches."""

AXES_HEIGHT = 2.0
"""Height of one panel, in inches (14 rows)."""

BOTTOM = 0.86
"""Bottom of the lower block's panels, in inches (its x label and the legend below)."""

BLOCK_GAP = 0.46
"""Between the bottom of the upper block's panels and the lower block's title, in inches
(tick labels and x label)."""

TITLE_DROP = 0.46
"""From a block's title to the top of its panels, in inches (the market titles)."""

TOP = 0.2
"""Above the upper block's title, in inches."""

X_LABEL_DROP = 0.28
"""From the bottom of a block's panels to the centre of its x label, in inches."""

LEGEND_X = 0.539
"""Centre of the legend, as a fraction of the width."""

LEGEND_Y = 0.19
"""Bottom of the legend, in inches."""

FIGURE_HEIGHT = BOTTOM + 2 * AXES_HEIGHT + 2 * TITLE_DROP + BLOCK_GAP + TOP
"""Height of the figure in inches."""


def _block_positions() -> tuple[tuple[float, float], tuple[float, float]]:
    """Return each block's title top and panel bottom, as fractions of the height."""
    lower_bottom = BOTTOM
    lower_title = lower_bottom + AXES_HEIGHT + TITLE_DROP
    upper_bottom = lower_title + BLOCK_GAP
    upper_title = upper_bottom + AXES_HEIGHT + TITLE_DROP
    return (
        (upper_title / FIGURE_HEIGHT, upper_bottom / FIGURE_HEIGHT),
        (lower_title / FIGURE_HEIGHT, lower_bottom / FIGURE_HEIGHT),
    )


@dataclass(frozen=True)
class CentreScaleLayout:
    """The place of every part of the figure and the panel title of each market.

    Positions are fractions of the figure (matplotlib's figure coordinates).
    """

    size: tuple[float, float]
    """Width and height in inches, before the crop."""
    left: float
    """Left edge of the first panel."""
    right: float
    """Right edge of the last panel."""
    gap: float
    """Space between two panels."""
    blocks: tuple[tuple[float, float], tuple[float, float]]
    """Title top and panel bottom of block (a), then of block (b)."""
    title_x: float
    """Left edge of a block title."""
    axes_height: float
    """Height of one panel."""
    x_label_drop: float
    """From a block's panel bottom to the centre of its x label."""
    legend_y: float
    """Bottom of the legend."""
    market_titles: dict[str, str]
    """The panel title of a market whose label is not its title."""


PAPER_LAYOUT = CentreScaleLayout(
    size=(FIGURE_WIDTH, FIGURE_HEIGHT),
    left=LEFT / FIGURE_WIDTH,
    right=1 - RIGHT / FIGURE_WIDTH,
    gap=GAP / FIGURE_WIDTH,
    blocks=_block_positions(),
    title_x=TITLE_X / FIGURE_WIDTH,
    axes_height=AXES_HEIGHT / FIGURE_HEIGHT,
    x_label_drop=X_LABEL_DROP / FIGURE_HEIGHT,
    legend_y=LEGEND_Y / FIGURE_HEIGHT,
    market_titles=MARKET_TITLES,
)
"""The paper's figure: the owner's layout narrowed to the text width, the stock market titled
"Stocks"."""

ORIGINAL_LAYOUT = CentreScaleLayout(
    size=(8.35, 7.3),
    left=0.166,
    right=0.975,
    gap=0.023,
    blocks=((0.938, 0.565), (0.500, 0.127)),
    title_x=0.034,
    axes_height=0.309,
    x_label_drop=0.039,
    legend_y=0.032,
    market_titles={},
)
"""The owner's own figure (``plot_rq2.py`` v3, 2026-11-19), number for number: 8.35 x 7.3 in
before the crop, "US stocks" as in ``rq2.csv``."""


def read_centre_scale(
    path: Path,
    *,
    models: tuple[tuple[str, str], ...] = RQ2_MODELS,
    markets: tuple[tuple[str, str], ...] = RQ2_MARKETS,
) -> CentreScaleRows:
    """Read ``rq2.csv`` as text, one row per model and market.

    Parameters
    ----------
    path : Path
        04-rq2-center-vs-scale ``tables/rq2.csv``.
    models : tuple of (str, str)
        The models expected (name, label); the paper's `RQ2_MODELS` by default.
    markets : tuple of (str, str)
        The markets expected (label, marker); the paper's `RQ2_MARKETS` by default.

    Returns
    -------
    dict of (str, str) to dict of str to str
        Every row, by ``(model, market)``; empty cells are empty strings.

    Raises
    ------
    ValueError
        If a row is not at horizon 1, or the table is not one row per model of `models` and
        market of `markets`.
    """
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if any(row["horizon"] != "1" for row in rows):
        raise ValueError("Expected the one-bar RQ2 table.")
    data = {(row["model"], row["market"]): row for row in rows}
    expected = {(model, market) for model, _ in models for market, _ in markets}
    if data.keys() != expected or len(rows) != len(expected):
        raise ValueError("Expected one estimate per model and market.")
    return data


def _signed_tick(x: float, _: object) -> str:
    """Label a tick with its sign: ``+10``, ``0``, and a true minus sign for ``\u221220``."""
    return f"+{x:g}" if x > 0 else f"{x:g}".replace("-", "\u2212")


def _plot_market(
    axis: "Axes",
    data: CentreScaleRows,
    metric: Metric,
    market_style: tuple[str, str],
    *,
    models: tuple[tuple[str, str], ...],
    market_titles: dict[str, str],
    show_model_labels: bool,
) -> None:
    """Draw one market's panel of one metric."""
    from matplotlib.ticker import FuncFormatter

    _, _, value_key, low_key, high_key, status_key, limits, ticks, reference = metric
    market, marker = market_style
    is_hit_rate = value_key == "hit_rate"

    for y, (model, _) in enumerate(models):
        if y % 2 == 0:
            axis.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        row = data[model, market]
        if not row[value_key]:
            axis.text(0.36, y, "—", color=REFERENCE, ha="center", va="center",
                      transform=axis.get_yaxis_transform(), fontsize=8)  # fmt: skip
            continue

        value = 100 * float(row[value_key])
        position = y - 0.10 if is_hit_rate else y
        fill = BLUE if row[status_key] == "True" else "white"
        if row[low_key] and row[high_key]:
            axis.hlines(position, 100 * float(row[low_key]), 100 * float(row[high_key]),
                        color=BLUE, lw=1.15, alpha=0.82, zorder=3)  # fmt: skip
        axis.plot(value, position, marker=marker, linestyle="none", ms=4.3,
                  mfc=fill, mec=BLUE, mew=1.0, zorder=4)  # fmt: skip

        if is_hit_rate and row["reference_hit_rate"]:
            axis.plot(
                100 * float(row["reference_hit_rate"]), y + 0.27,
                marker="|", linestyle="none", color=INK, ms=4.0, mew=0.9, zorder=4,
            )  # fmt: skip

    axis.axvline(reference, color=REFERENCE, lw=0.85, zorder=2)
    axis.set(xlim=limits, ylim=(len(models) - 0.5, -0.5), xticks=ticks,
             yticks=range(len(models)))  # fmt: skip
    if not is_hit_rate:
        axis.xaxis.set_major_formatter(FuncFormatter(_signed_tick))
    axis.set_yticklabels([label for _, label in models] if show_model_labels else [])
    axis.tick_params(axis="y", length=0, pad=6)
    axis.tick_params(axis="x", length=2.8, pad=3)
    axis.set_title(
        market_titles.get(market, market), color="black", fontweight="bold", fontsize=9.0, pad=7
    )


def draw_centre_scale(
    data: CentreScaleRows,
    *,
    pdf_path: Path,
    png_path: Path,
    models: tuple[tuple[str, str], ...] = RQ2_MODELS,
    markets: tuple[tuple[str, str], ...] = RQ2_MARKETS,
    layout: CentreScaleLayout = PAPER_LAYOUT,
) -> None:
    """Draw the figure of RQ2: (a) the sign hit rate, (b) the coverage minus 80 %, by market.

    One panel per market of `markets` in each block, one row per model of `models`, every
    second row on a light band; blue marks with their 95 % intervals; filled: centre-tested
    (a) or calibrated (b); a dash where there is no estimate; in (a) a dark tick at
    climatology's hit rate. The figure is drawn at the layout's size and cropped to its content
    (0.04 in of padding); matplotlib's settings are restored afterwards.

    Parameters
    ----------
    data : dict of (str, str) to dict of str to str
        `read_centre_scale` of ``rq2.csv``.
    pdf_path : Path
        The PDF to write (no creation date, so a rerun gives the same file).
    png_path : Path
        The PNG to write, at 320 dpi.
    models : tuple of (str, str)
        The rows, top to bottom (name, label); the paper's `RQ2_MODELS` by default.
    markets : tuple of (str, str)
        The panels, left to right (label, marker); the paper's `RQ2_MARKETS` by default.
    layout : CentreScaleLayout
        `PAPER_LAYOUT` (7.0 in wide, the default) or `ORIGINAL_LAYOUT` (the owner's script).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with mpl.rc_context():
        plt.rcParams.update(serif_rc())
        plt.rcParams.update({
            "font.size": 9.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#A6B1BE",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": "#46566A",
            "ytick.color": INK,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
        })  # fmt: skip
        figure = plt.figure(figsize=layout.size, facecolor="white")
        left, right, gap = layout.left, layout.right, layout.gap
        width = (right - left - (len(markets) - 1) * gap) / len(markets)
        for metric, (title_y, bottom) in zip(METRICS, layout.blocks, strict=True):
            title, x_label = metric[:2]
            figure.text(layout.title_x, title_y, title, va="top", fontweight="bold",
                        fontsize=12.2)  # fmt: skip
            for j, market in enumerate(markets):
                axis = figure.add_axes(
                    (left + j * (width + gap), bottom, width, layout.axes_height)
                )
                _plot_market(
                    axis, data, metric, market, models=models,
                    market_titles=layout.market_titles, show_model_labels=(j == 0),
                )  # fmt: skip
            figure.text((left + right) / 2, bottom - layout.x_label_drop, x_label,
                        ha="center", va="center")  # fmt: skip

        guide = [
            Line2D([], [], marker="|", linestyle="none", color=INK, ms=5.8,
                   label="Climatology (a)"),
            Line2D([], [], marker="o", linestyle="none", color=INK, mfc="white", ms=4.4,
                   label="Untested (a)"),
            Line2D([], [], marker="o", linestyle="none", color=INK, mfc="white", ms=4.4,
                   label="Coverage criterion unmet (b)"),
            Line2D([], [], marker="_", linestyle="none", color=REFERENCE, ms=6,
                   label="No estimate"),
        ]  # fmt: skip
        anchor = (LEGEND_X, layout.legend_y)
        figure.legend(handles=guide, loc="lower center", bbox_to_anchor=anchor,
                      frameon=False, ncols=4, fontsize=7.9, handlelength=1.05,
                      columnspacing=1.45, handletextpad=0.45)  # fmt: skip
        for path in (png_path, pdf_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # cropped to the drawn content, so the paper's file is at most FIGURE_WIDTH wide
        figure.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=CROP_PAD)
        figure.savefig(
            pdf_path,
            dpi=320,
            bbox_inches="tight",
            pad_inches=CROP_PAD,
            metadata={"CreationDate": None, "Creator": None},
        )
        plt.close(figure)
