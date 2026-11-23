"""The paper's power figure (``fig:power``): every cell's outcome and the testable cells' CDF.

The owner's plot_mde_coverage.py (fourth version, 2026-11-19) in its colours, layout and labels,
redrawn for the final paper (the owner's final pass, 2026-11-19): drawn at the text width
(7.0 in) and included unscaled, so every text is at least 7 pt in the paper, in TeX Gyre
Termes; the data come from the cells table here instead of his CSV reader, and the figure is
saved through `save_figure` (checks, no dates). (a) one 100 % bar per horizon over all
its cells, split into MDE at most the chosen X, MDE above X but reached within the largest
gain planted, MDE not reached by it, and cells the rule cannot test, with the rule's share
among the testable cells and the cells not reached in a table key inside (b); (b) the step
ECDF of the testable cells' MDEs per horizon, whose plateau below 100 % is the cells not
reached. Both panels get the same width, labels included.

The owner chose text of 7 to 9 pt, below the AAAI-27 kit's 9 pt (charts.md 2a.14).
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.figures._style import save_figure, serif_rc

if TYPE_CHECKING:
    from matplotlib.axes import Axes

HORIZONS = (
    (1, "1 bar", "#1669A5", 3.05),
    (4, "4 bars", "#DD9527", 2.10),
    (24, "24 bars", "#BD749C", 1.45),
)
"""Horizon, name, colour and line width (the owner's)."""

FIGURE_SIZE = (7.0, 2.54)
"""Width and height in inches: the paper's text width, so the figure is included unscaled."""

AXES_BOTTOM = 0.80
"""Bottom of both panels' axes, in inches (room for the x label and the key of (a))."""

AXES_TOP = 2.14
"""Top of both panels' axes, in inches (room for the titles above)."""

PANEL_GAP_PT = 22.0
"""Distance between the outer bounds of the two panels, in points."""

INK = "#172638"
SECONDARY = "#46566A"
GRID = "#E3E9EF"
MIN_QUARTERS = 3
"""The owner's colours and the rule's fewest test quarters (shown in the labels)."""

SMALLEST_TEXT = 7.0
"""The smallest font size of the figure, in points in the paper (the owner's floor)."""

TICK_SIZE = 7.5
"""Size in points of tick labels, the counts in the bars, the keys and the cell counts."""

LABEL_SIZE = 8.0
"""Size in points of axis labels, horizon names and subtitles."""

TITLE_SIZE = 9.0
"""Size in points of the panel titles."""

RC: dict[Any, Any] = {
    # the serif (TeX Gyre Termes) comes from `serif_rc`
    "font.size": LABEL_SIZE,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9BA8B6",
    "axes.linewidth": 0.6,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "figure.constrained_layout.use": False,
    "savefig.bbox": "standard",
}
"""The owner's matplotlib settings (TrueType fonts in the PDF, never Type 3)."""


@dataclass(frozen=True)
class PowerLines:
    """The rule's numbers that the power figure draws as lines and limits.

    Attributes
    ----------
    chosen_pct : float
        The chosen threshold X in %, a value of the shares table's ``threshold_pct``.
    required_share : float
        The share of the testable cells the rule needs, between 0 and 1.
    x_max : float
        The largest planted gain tried and the right end of the x axis of panel (b), in %.
    power : float
        The power at which each cell's MDE is found, between 0 and 1 (03's ``power_target``).
    """

    chosen_pct: float
    required_share: float
    x_max: float
    power: float = 0.8


def empirical_shares(cells: pd.DataFrame, horizon: int, grid: np.ndarray) -> np.ndarray:
    """Return the share of a horizon's cells with a reached MDE at most each value of `grid`.

    The denominator is every cell of the horizon given, a cell whose MDE was never reached
    included (it is never resolved), as in `src.benchmark.threshold_shares`.

    Parameters
    ----------
    cells : pd.DataFrame
        The cells: ``horizon``, ``mde_pct`` and ``is_reached``.
    horizon : int
        The horizon in bars.
    grid : np.ndarray
        The thresholds X in %, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        The shares between 0 and 1, shape ``(n,)``.

    Raises
    ------
    ValueError
        If `cells` has no cell of `horizon`.
    """
    own = cells[cells["horizon"].astype(int) == horizon]
    if own.empty:
        raise ValueError(f"no cell of horizon {horizon}")
    reached = np.sort(own.loc[own["is_reached"].astype(bool), "mde_pct"].to_numpy(dtype=float))
    return np.searchsorted(reached, grid, side="right") / len(own)


def outcome_counts(cells: pd.DataFrame, chosen_pct: float) -> pd.DataFrame:
    """Count each horizon's cells by outcome at the chosen threshold.

    Parameters
    ----------
    cells : pd.DataFrame
        Every scored cell: ``horizon``, ``mde_pct``, ``is_reached`` and ``is_testable`` (the
        cell has the test quarters the rule needs, `src.benchmark.detectable_cells`).
    chosen_pct : float
        The threshold X in %.

    Returns
    -------
    pd.DataFrame
        One row per horizon, increasing: ``horizon``, ``cells`` (all), ``testable``, and the
        counts of `OUTCOMES`: ``resolved`` (testable, MDE reached and at most X), ``reached``
        (testable, reached above X), ``not_reached`` (testable, not reached within the gains
        tried) and ``untestable``; the four add up to ``cells``.
    """
    rows: list[dict[str, int]] = []
    for horizon, own in cells.groupby(cells["horizon"].astype(int)):
        testable = own["is_testable"].astype(bool).to_numpy()
        reached = own["is_reached"].astype(bool).to_numpy()
        small = own["mde_pct"].to_numpy(dtype=float) <= chosen_pct
        rows.append({
            "horizon": int(horizon), "cells": len(own), "testable": int(testable.sum()),
            "resolved": int((testable & reached & small).sum()),
            "reached": int((testable & reached & ~small).sum()),
            "not_reached": int((testable & ~reached).sum()),
            "untestable": int((~testable).sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def draw_power_summary(
    cells: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    lines: PowerLines,
    pdf_path: Path,
    grey_path: Path,
    exports: Sequence[Path] = (),
) -> None:
    """Draw the owner's two-panel power figure from every scored cell.

    Parameters
    ----------
    cells : pd.DataFrame
        Every scored cell (``tables/mde.csv``): ``horizon``, ``mde_pct``, ``is_reached``,
        ``is_testable`` and ``model``.
    shares : pd.DataFrame
        The table of `src.benchmark.threshold_shares` of the testable cells
        (``tables/mde_threshold.csv``), checked against the testable cells at every
        threshold before drawing.
    lines : PowerLines
        The chosen threshold, the required share and the largest gain planted.
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.
    exports : sequence of Path
        Further copies (SVG, 320 dpi PNG).

    Raises
    ------
    ValueError
        If a horizon is not 1, 4 or 24 bars or has no testable cell, the chosen threshold is
        not one of `shares`, the testable cells disagree with `shares`, or a reached MDE lies
        outside 0 to the largest gain.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    grid = shares["threshold_pct"].to_numpy(dtype=float)
    if not np.isclose(grid, lines.chosen_pct).any():
        raise ValueError(f"chosen threshold {lines.chosen_pct} is not a threshold of the table")
    known = {horizon for horizon, *_ in HORIZONS}
    if set(cells["horizon"].astype(int)) - known:
        raise ValueError("the cells contain horizons outside 1, 4 and 24 bars")
    testable_cells = cells[cells["is_testable"].astype(bool)]
    for horizon, *_ in HORIZONS:
        expected = shares[f"share_{horizon}h"].to_numpy(dtype=float)
        if not np.allclose(empirical_shares(testable_cells, horizon, grid), expected):
            raise ValueError(f"the testable cells differ from the table at {horizon} bars")
    groups, all_kronos = _groups(cells, lines)
    height = FIGURE_SIZE[1]
    bottom, top = AXES_BOTTOM / height, AXES_TOP / height
    with mpl.rc_context({**serif_rc(), **RC}):
        figure = plt.figure(figsize=FIGURE_SIZE)
        # measured with Agg whatever the backend, so the panel balance sees the text sizes of
        # the saved files (the PDF renderer measures text at another resolution)
        FigureCanvasAgg(figure)
        coverage = figure.add_axes((0.075, bottom, 0.41, top - bottom))
        cumulative = figure.add_axes((0.56, bottom, 0.425, top - bottom))
        _plot_coverage(coverage, groups, lines, all_kronos=all_kronos)
        _plot_cumulative(cumulative, groups, lines)
        _balance_panels(figure, coverage, cumulative, PANEL_GAP_PT)
        cell_count = sum(group[0] for group in groups.values())
        testable_count = sum(group[1] for group in groups.values())
        power, target = f"{100 * lines.power:g}%", f"{100 * lines.required_share:g}%"
        for axis, title, subtitle in (
            (coverage, "(a) MDE outcomes by horizon",
             f"All {cell_count} combinations; MDE: smallest gain found with {power} power"),
            (cumulative, "(b) Cumulative MDE by horizon",
             f"{testable_count} testable combinations; "
             f"prespecified 1-bar target: {target} of them"),
        ):  # fmt: skip
            bounds = axis.get_position()
            x = bounds.x0 + bounds.width / 2
            figure.text(x, bounds.y1 + 0.36 / height, title, fontweight="bold",
                        fontsize=TITLE_SIZE, va="top", ha="center")  # fmt: skip
            figure.text(x, bounds.y1 + 0.20 / height, subtitle, fontsize=LABEL_SIZE,
                        color=SECONDARY, va="top", ha="center")  # fmt: skip
        save_figure(
            figure, pdf_path, grey_path, min_font_size=SMALLEST_TEXT, exports=exports,
            crop=0.01,
        )  # fmt: skip


def _groups(
    cells: pd.DataFrame, lines: PowerLines
) -> tuple[dict[int, tuple[int, int, np.ndarray, tuple[int, int, int, int]]], bool]:
    # the owner's load_groups on the cells table: per horizon, all cells, testable cells, the
    # sorted reached MDEs and the four outcome counts; and whether every unreached testable
    # cell is a Kronos one
    groups: dict[int, tuple[int, int, np.ndarray, tuple[int, int, int, int]]] = {}
    unresolved: set[str] = set()
    for horizon, *_ in HORIZONS:
        own = cells[cells["horizon"].astype(int) == horizon]
        testable = own[own["is_testable"].astype(bool)]
        if testable.empty:
            raise ValueError(f"no testable cells for the {horizon}-bar horizon")
        reached_mask = testable["is_reached"].astype(bool)
        unresolved |= set(testable.loc[~reached_mask, "model"].astype(str))
        mde = np.sort(testable.loc[reached_mask, "mde_pct"].to_numpy(dtype=float))
        if ((mde < 0) | (mde > lines.x_max)).any():
            raise ValueError(f"a reached MDE lies outside 0 to {lines.x_max}%")
        low = int(np.count_nonzero(mde <= lines.chosen_pct))
        counts = (low, len(mde) - low, len(testable) - len(mde), len(own) - len(testable))
        groups[horizon] = (len(own), len(testable), mde, counts)
    all_kronos = bool(unresolved) and all(m.lower().startswith("kronos") for m in unresolved)
    return groups, all_kronos


def _plot_coverage(
    ax: "Axes",
    groups: dict[int, tuple[int, int, np.ndarray, tuple[int, int, int, int]]],
    lines: PowerLines,
    *,
    all_kronos: bool,
) -> None:
    # the owner's plot_coverage, unchanged but for the threshold and limits from `lines`
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    threshold, limit = lines.chosen_pct, lines.x_max
    categories = (
        (f"MDE ≤ {threshold:g}%", "#167089", "white", None),
        (f"{threshold:g}% < MDE ≤ {limit:g}%", "#9CCFDF", INK, None),
        ("Not reached within tested range" + (" (all Kronos)" if all_kronos else ""),
         "#F2BC6D", "#462A0D", None),
        (f"Not testable (<{MIN_QUARTERS} test quarters)", "#E5E9EE", INK, "///"),
    )  # fmt: skip
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=GRID, linewidth=0.6)
    ax.set(xlim=(0, 100), ylim=(-0.42, 2.86), xticks=np.arange(0, 101, 25))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Share of all combinations in each horizon", labelpad=3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    for y, (horizon, name, *_) in zip((2, 1, 0), HORIZONS, strict=True):
        total, _, _, counts = groups[horizon]
        ax.text(-0.025, y + 0.11, name, transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontweight="bold")  # fmt: skip
        ax.text(-0.025, y - 0.16, f"n = {total}", transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=TICK_SIZE, color=SECONDARY)  # fmt: skip
        left = 0.0
        for count, (_, fill, fg, hatch) in zip(counts, categories, strict=True):
            if count == 0:
                continue
            share = 100 * count / total
            center = left + share / 2
            ax.barh(y, share, left=left, height=0.52, color=fill, hatch=hatch,
                    edgecolor="white", linewidth=0.6, zorder=3)  # fmt: skip
            # a segment too narrow for its two lines is labelled above, on a leader line
            if share < 12:
                ax.annotate(
                    f"{count} ({share:.1f}%)", xy=(center, y + 0.27),
                    xytext=(center, y + 0.42), ha="center", va="bottom",
                    fontsize=TICK_SIZE, color=INK, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.6,
                                "shrinkA": 0, "shrinkB": 0},
                )  # fmt: skip
            else:
                ax.text(center, y, f"{count}\n{share:.1f}%", color=fg,
                        ha="center", va="center",
                        fontsize=SMALLEST_TEXT if share < 15 else TICK_SIZE,
                        linespacing=1.0)  # fmt: skip
            left += share
    handles = [
        Patch(facecolor=fill, edgecolor="white", hatch=hatch, label=label)
        for label, fill, _, hatch in categories
    ]
    # matplotlib fills the key column by column: the two MDE outcomes in the first column and
    # the two long labels in the second, so the key is no wider than the panel
    ax.legend(handles=handles, ncols=2, frameon=False,
              loc="upper left", bbox_to_anchor=(-0.015, -0.25),
              borderaxespad=0, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.9, labelspacing=0.35, fontsize=SMALLEST_TEXT)  # fmt: skip


def _add_cumulative_legend(
    ax: "Axes",
    groups: dict[int, tuple[int, int, np.ndarray, tuple[int, int, int, int]]],
    lines: PowerLines,
) -> None:
    # the owner's table key inside panel (b): horizon, share at X with counts, combinations
    # not reached
    threshold, target = lines.chosen_pct, lines.required_share * 100
    legend = ax.inset_axes((0.33, 0.05, 0.67, 0.45), facecolor="white", zorder=5)
    legend.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for spine in legend.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    legend.text(0.04, 0.80, "Horizon", va="center", fontsize=SMALLEST_TEXT, fontweight="bold")
    legend.text(0.545, 0.88, f"At {threshold:g}%", ha="center", va="center",
                fontsize=SMALLEST_TEXT, fontweight="bold")  # fmt: skip
    legend.text(0.545, 0.72, f"target {target:g}%", ha="center", va="center",
                fontsize=SMALLEST_TEXT, color=SECONDARY)  # fmt: skip
    legend.text(0.845, 0.88, "Not reached", ha="center", va="center",
                fontsize=SMALLEST_TEXT, fontweight="bold")  # fmt: skip
    legend.text(0.845, 0.72, "combinations", ha="center", va="center",
                fontsize=SMALLEST_TEXT, color=SECONDARY)  # fmt: skip
    legend.hlines(0.63, 0.04, 0.96, color=GRID, linewidth=0.6)
    for y, (horizon, name, color, width) in zip((0.49, 0.30, 0.11), HORIZONS, strict=True):
        _, testable, _, counts = groups[horizon]
        legend.plot([0.04, 0.095, 0.15], [y] * 3, color=color, lw=width * 0.6,
                    marker="o", markevery=[1], ms=3.0)  # fmt: skip
        legend.text(0.165, y, name, va="center", fontsize=TICK_SIZE)
        legend.text(0.545, y, f"{100 * counts[0] / testable:.1f}% ({counts[0]}/{testable})",
                    ha="center", va="center", fontsize=TICK_SIZE)  # fmt: skip
        legend.text(0.845, y, f"{counts[2]}", ha="center", va="center",
                    fontsize=TICK_SIZE)  # fmt: skip


def _plot_cumulative(
    ax: "Axes",
    groups: dict[int, tuple[int, int, np.ndarray, tuple[int, int, int, int]]],
    lines: PowerLines,
) -> None:
    # the owner's plot_cumulative, unchanged but for the threshold and limits from `lines`
    from matplotlib.ticker import PercentFormatter

    threshold, limit, target = lines.chosen_pct, lines.x_max, lines.required_share * 100
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.6)
    ax.axvline(threshold, color="#A0ADBB", lw=0.7, zorder=2)
    ax.axhline(target, color="#A0ADBB", lw=0.7, zorder=2)
    for horizon, _, color, width in HORIZONS:
        _, testable, mde, counts = groups[horizon]
        unique, event_counts = np.unique(mde, return_counts=True)
        share = 100 * np.cumsum(event_counts) / testable
        # the wider blue stroke first, so it stays visible around the thinner ones where
        # their empirical CDFs take exactly the same values
        ax.step(np.r_[0, unique, limit], np.r_[0, share, 100 * len(mde) / testable],
                where="post", color=color, linestyle="-", linewidth=width * 0.6,
                zorder=3)  # fmt: skip
        ax.plot(threshold, 100 * counts[0] / testable, marker="o", ms=3.0,
                mfc=color, mec="white", mew=0.4, linestyle="none", zorder=4)  # fmt: skip
    ax.set(xlim=(0, limit), ylim=(0, 100), xticks=np.arange(0, limit + 1, 2),
           yticks=np.arange(0, 101, 20))  # fmt: skip
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    for axis, value in ((ax.xaxis, threshold), (ax.yaxis, target)):
        for position, label in zip(axis.get_ticklocs(), axis.get_ticklabels(), strict=True):
            if np.isclose(position, value):
                label.set_fontweight("bold")
    apostrophe = "\N{RIGHT SINGLE QUOTATION MARK}"
    ax.set_xlabel(f"MDE threshold X (% of the selected baseline{apostrophe}s CRPS)", labelpad=3)
    ax.set_ylabel("Testable combinations with MDE ≤ X", labelpad=3)
    _add_cumulative_legend(ax, groups, lines)


def _balance_panels(figure: "Any", left_ax: "Axes", right_ax: "Axes", gap_pt: float) -> None:
    # the owner's balance_panels: both panels the same width, tick and axis labels included
    to_figure = figure.transFigure.inverted()
    gap = gap_pt / (72 * figure.get_figwidth())
    panels = (left_ax, right_ax)
    figure.draw_without_rendering()
    left = _content(left_ax).transformed(to_figure).x0
    right = _content(right_ax).transformed(to_figure).x1
    panel_width = (right - left - gap) / 2
    # measured four times: the row labels left of (a) sit at an offset proportional to its width
    for _ in range(4):
        for axis, panel_left in zip(panels, (left, left + panel_width + gap), strict=True):
            bounds = axis.get_position()
            content = _content(axis).transformed(to_figure)
            left_overhang = bounds.x0 - content.x0
            right_overhang = content.x1 - bounds.x1
            axis.set_position((panel_left + left_overhang, bounds.y0,
                               panel_width - left_overhang - right_overhang,
                               bounds.height))  # fmt: skip
        figure.draw_without_rendering()


def _content(axis: "Axes") -> "Any":
    # the box of an axes with its tick and axis labels, in display units
    box = axis.get_tightbbox()
    if box is None:
        raise ValueError("an axes of the figure has no drawn content")
    return box
