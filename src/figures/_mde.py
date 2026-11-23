"""Figures of the power analysis: the MDE of every cell and the share of cells a threshold resolves.

Drawn from the tables of `src.benchmark._power` (``tables/mde.csv`` and
``tables/mde_threshold.csv``) by the analyses of 03-rq1-null and 09-mde-threshold.
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
    HORIZON_LINESTYLE,
    REFERENCE_LINE,
    SINGLE_WIDTH,
    save_figure,
    use_paper_style,
)


def draw_mde(mde: pd.DataFrame, *, threshold_pct: float, pdf_path: Path, grey_path: Path) -> None:
    """Draw the MDE of every cell per market, one panel per horizon, with the threshold line.

    Each model is one dot in the row of its market. A cell whose MDE was never reached is an
    open triangle at the largest gain tried. The figure is 6.80 x 2.3 in, in the paper style.

    Parameters
    ----------
    mde : pd.DataFrame
        The table of `minimum_detectable_effects`: ``market``, ``horizon``, ``mde_pct``,
        ``is_reached`` and ``largest_pct``.
    threshold_pct : float
        The threshold X in % of the baseline's CRPS, drawn as a vertical dashed line (not
        drawn when NaN).
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    markets = sorted(set(mde["market"].tolist()))
    horizons = sorted(set(mde["horizon"].tolist()))
    top = float(mde["largest_pct"].max())
    rows = mde["market"].map(markets.index)
    # the paper style is set only inside this block, so the caller's settings stay unchanged
    with mpl.rc_context():
        use_paper_style()
        figure, axes = plt.subplots(
            1, len(horizons), figsize=(DOUBLE_WIDTH, 2.3), sharex=True, sharey=True, squeeze=False
        )
        for axis, horizon in zip(axes[0], horizons, strict=True):
            reached = (mde["horizon"] == horizon) & mde["is_reached"]
            never = (mde["horizon"] == horizon) & ~mde["is_reached"]
            axis.plot(
                mde.loc[reached, "mde_pct"],
                rows[reached],
                "o",
                color="black",
                label="MDE of a model",
            )
            axis.plot(
                np.full(int(never.sum()), top),
                rows[never],
                "^",
                color="black",
                markerfacecolor="none",
                label=f"never reached (> {top:g} %)",
            )
            if not np.isnan(threshold_pct):
                axis.axvline(
                    threshold_pct, **REFERENCE_LINE, label=f"threshold {threshold_pct:g} %"
                )
            axis.set_title(_bars(horizon))
        axes[0][0].set_yticks(range(len(markets)), markets)
        axes[0][0].set_ylim(len(markets) - 0.5, -0.5)
        axes[0][0].set_xlim(0, top * 1.08)
        figure.legend(handles=axes[0][0].get_lines(), loc="outside upper center", ncols=3)
        figure.supxlabel("MDE (% of the best baseline's CRPS)")
        save_figure(figure, pdf_path, grey_path)


def draw_threshold_shares(
    shares: pd.DataFrame,
    *,
    chosen_pct: float,
    pdf_path: Path,
    grey_path: Path,
    required_share: float = float("nan"),
    judged_horizon: int | None = None,
) -> None:
    """Draw the share of cells resolved at each threshold, one line style per horizon.

    The figure is 3.15 x 2.0 in, in the paper style: black lines in the line style of
    `HORIZON_LINESTYLE`, all horizons together as a grey dash-dotted line, the chosen threshold
    as a dashed line named at its top, the legend in two columns at the lower right.

    With `judged_horizon` (the horizon the rule judges, charts.md 2a.3) that horizon's line is
    black and thicker, the other horizons grey, the all-horizons line is left out and the
    judged horizon's share at the chosen threshold is marked and labelled; with
    `required_share` the share the rule needs is a dotted line named at its right end
    (charts.md 2a.7). The y axis is then in percent.

    Parameters
    ----------
    shares : pd.DataFrame
        The table of `threshold_shares`: ``threshold_pct``, ``share`` and one
        ``share_<horizon>h`` column per horizon.
    chosen_pct : float
        The chosen threshold, marked with a vertical dashed line (not drawn when NaN).
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.
    required_share : float
        The share of cells the rule needs, drawn as a dotted line (not drawn when NaN).
    judged_horizon : int or None
        The horizon the rule judges, drawn in front; None draws every horizon alike.

    Raises
    ------
    ValueError
        If a horizon has no line style in `HORIZON_LINESTYLE`, or `judged_horizon` is not a
        horizon of `shares`.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter

    columns = [column for column in shares.columns if column.startswith("share_")]
    horizons = [int(column.removeprefix("share_").removesuffix("h")) for column in columns]
    for horizon in horizons:
        if horizon not in HORIZON_LINESTYLE:
            raise ValueError(
                f"horizon {horizon} has no line style; expected one of {sorted(HORIZON_LINESTYLE)}"
            )
    if judged_horizon is not None and judged_horizon not in horizons:
        raise ValueError(f"judged horizon {judged_horizon} is not one of {horizons}")
    scale = 1.0 if judged_horizon is None else 100.0
    # the paper style is set only inside this block, so the caller's settings stay unchanged
    with mpl.rc_context():
        use_paper_style()
        figure, axis = plt.subplots(figsize=(SINGLE_WIDTH, 2.0))
        for column, horizon in zip(columns, horizons, strict=True):
            is_judged = horizon == judged_horizon
            colour = "black" if judged_horizon is None or is_judged else "#8C8C8C"
            axis.plot(
                shares["threshold_pct"],
                shares[column] * scale,
                color=colour,
                linestyle=HORIZON_LINESTYLE[horizon],
                linewidth=1.6 if is_judged else 1.1,
                label=_bars(horizon),
                zorder=3 if is_judged else 2,
            )
        if judged_horizon is None:
            axis.plot(
                shares["threshold_pct"],
                shares["share"],
                color="#888888",
                linestyle="-.",
                label="all horizons",
            )
        if not np.isnan(chosen_pct):
            # the reference line is named at its top instead of in the legend (charts.md 3.6)
            axis.axvline(chosen_pct, **REFERENCE_LINE)
            axis.annotate(
                f"X = {chosen_pct:g} %",
                xy=(chosen_pct, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                color="#333333",
            )
        _mark_requirement(axis, shares, chosen_pct, required_share, judged_horizon, scale)
        axis.set_xlabel("Threshold X (% of the best baseline's CRPS)")
        if judged_horizon is None:
            axis.set_ylabel(r"Share of cells with MDE $\leq$ X")
            axis.set_ylim(0, 1.02)
            axis.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))
        else:
            axis.set_ylabel(r"Cells with MDE $\leq$ X (%)")
            axis.set_ylim(0, 102)
            axis.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        # two columns in the empty lower-right corner, below the plateau of the lines
        axis.legend(loc="lower right", ncols=2, handlelength=2.2, columnspacing=1.0)
        save_figure(figure, pdf_path, grey_path)


def _mark_requirement(
    axis: "Axes",
    shares: pd.DataFrame,
    chosen_pct: float,
    required_share: float,
    judged_horizon: int | None,
    scale: float,
) -> None:
    # the share the rule needs as a named dotted line, and the judged horizon's share at the
    # chosen threshold as a labelled point (charts.md 2a.7)
    if not np.isnan(required_share):
        axis.axhline(required_share * scale, color="#333333", linestyle=":", linewidth=0.8)
        axis.annotate(
            f"required: {required_share * scale:g} %",
            xy=(1.0, required_share * scale),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="#333333",
        )
    if judged_horizon is not None and not np.isnan(chosen_pct):
        column = f"share_{judged_horizon}h"
        at = shares.loc[np.isclose(shares["threshold_pct"], chosen_pct), column]
        if not at.empty:
            value = float(at.iloc[0]) * scale
            axis.plot(chosen_pct, value, "o", color="black", markersize=4, zorder=4)
            axis.annotate(
                f"{value:.0f} %",
                xy=(chosen_pct, value),
                xytext=(5, -3),
                textcoords="offset points",
                ha="left",
                va="top",
                color="black",
            )


def _bars(horizon: int) -> str:
    # a horizon counts the asset's own bars, not calendar hours (Section 3.2 of the paper)
    return f"{horizon} bar" if horizon == 1 else f"{horizon} bars"
