"""Draw the paper's RQ4 figure (fig:rq4): one-bar pre/post-release CRPS DiD estimates and intervals.

Reads the committed ``tables/verdict.csv`` of 06-rq4-leakage's test run (the ``leakage`` test at
1 bar, before against after the release) and draws the owner's layout (2026-11-18): panel (a)
the eleven other checkpoints, panel (b) the two Kronos checkpoints, one column per market,
every interval as reported in the table (x 100, in target units). Writes
``paper/figures/fig_rq4.pdf`` and ``paper/figures/fig_rq4.png``. Run from the lab root:
``.venv/bin/python paper/scripts/make_fig_rq4.py``.
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

import argparse
import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.figures import serif_rc

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type Estimates = dict[tuple[str, str], tuple[float, float, float]]
"""The DiD, lower and upper bound in target units x 100, by model and market."""

VERDICT = Path("experiments/06-rq4-leakage/results/test-2026-11-18/tables/verdict.csv")
"""The committed test verdict of 06-rq4-leakage."""

FIGURE = Path("paper/figures/fig_rq4")
"""The base name of the paper's figure (``.pdf`` and ``.png``)."""

MODELS = (
    ("chronos-bolt-small", "Chronos-Bolt (S)"),
    ("chronos-bolt-base", "Chronos-Bolt (B)"),
    ("tirex-1.1", "TiRex"),
    ("moirai-2-small", "Moirai 2 (S)"),
    ("timesfm-2.5", "TimesFM 2.5"),
    ("chronos-2", "Chronos-2"),
    ("toto-2-22m", "Toto 2 (22M)"),
    ("toto-2-313m", "Toto 2 (313M)"),
    ("tirex-2", "TiRex-2"),
    ("timesfm-3", "TimesFM 3"),
    ("t0-gamma", "t0"),
)
"""The models of panel (a), top to bottom, with their row labels."""

KRONOS = (("kronos-small", "Kronos (S)"), ("kronos-base", "Kronos (B)"))
"""The models of panel (b), top to bottom, with their row labels."""

MARKETS = (
    ("crypto", "Crypto"),
    ("forex", "FX"),
    ("commodity", "Commodities"),
    ("index", "Indices"),
    ("stock", "Stocks"),
)
"""The markets, left to right, with their column titles."""

BLUE = "#1669A5"
INK = "#172638"
AXIS = "#8290A0"
STRIPE = "#F2F5F8"
MINUS = "\u2212"
"""The minus sign of the ticks and the x label."""


def load_estimates(source: Path) -> Estimates:
    """Read the 1-bar before/after leakage estimates of every model and market.

    Parameters
    ----------
    source : Path
        A verdict table of 06-rq4-leakage (``tables/verdict.csv``).

    Returns
    -------
    Estimates
        ``did``, ``low`` and ``high`` x 100 by (model, market), for the 13 models and 5 markets.

    Raises
    ------
    ValueError
        If a cell is repeated or not scored, its interval does not hold its estimate, its
        estimate is not ``skill_first - skill_second``, or a model or market is missing or extra.
    """
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["test"] == "leakage" and row["horizon"] == "1"
            and row["first"] == "before" and row["second"] == "after"
        ]  # fmt: skip
    estimates: Estimates = {}
    for row in rows:
        key = row["model"], row["market"]
        if key in estimates or row["status"] != "scored":
            raise ValueError(f"Expected one scored result for {key}.")
        estimate, lower, upper = (float(row[field]) for field in ("did", "low", "high"))
        contrast = float(row["skill_first"]) - float(row["skill_second"])
        if not lower <= estimate <= upper or not math.isclose(contrast, estimate, abs_tol=1e-12):
            raise ValueError(f"Invalid estimate or interval for {key}.")
        estimates[key] = (100 * estimate, 100 * lower, 100 * upper)
    expected = {(model, market) for model, _ in MODELS + KRONOS for market, _ in MARKETS}
    if estimates.keys() != expected:
        raise ValueError("Expected results for 13 models across all five markets.")
    return estimates


def signed_tick(value: float, _position: float) -> str:
    """Write a tick with its sign (``+1``, minus sign and ``1``) and ``0`` without one."""
    if value == 0:
        return "0"
    return ("+" if value > 0 else MINUS) + f"{abs(value):g}"


def plot_market(
    ax: "Axes",
    estimates: Estimates,
    market: tuple[str, str],
    models: tuple[tuple[str, str], ...],
    show_model_labels: bool,
    x_axis: tuple[float, tuple[int, ...]],
) -> None:
    """Draw one market's column: one mark and interval per model, on striped rows.

    `x_axis` is the half-width of the x-axis and its ticks.

    Raises
    ------
    ValueError
        If an interval reaches the axis limit (it would be clipped).
    """
    market_id, market_name = market
    limit, ticks = x_axis
    for y, (model_id, _) in enumerate(models):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=STRIPE, lw=0, zorder=0)
        estimate, lower, upper = estimates[model_id, market_id]
        if lower <= -limit or upper >= limit:
            raise ValueError(f"Plot limit would clip the interval for {model_id}, {market_id}.")
        ax.errorbar(estimate, y, xerr=[[estimate - lower], [upper - estimate]],
                    fmt="o", color=BLUE, mfc="white", mec=BLUE, mew=1.05,
                    ms=4.5, elinewidth=1.15, capsize=2.0, zorder=3)  # fmt: skip
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    ax.set(xlim=(-limit, limit), ylim=(len(models) - 0.5, -0.5),
           xticks=ticks, yticks=range(len(models)))  # fmt: skip
    ax.xaxis.set_major_formatter(FuncFormatter(signed_tick))
    ax.set_yticklabels([label for _, label in models] if show_model_labels else [])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=2.8, pad=4)
    ax.set_title(market_name, color="black", fontsize=9.0, fontweight="bold", pad=6)


def draw(estimates: Estimates, output: Path) -> None:
    """Draw the figure and write it as ``output`` with the suffixes ``.png`` and ``.pdf``.

    The figure is drawn at 8.35 x 5.25 in and saved at 320 dpi, cropped to its content with 0.01 in
    of padding; matplotlib's settings stay changed.
    """
    plt.rcParams.update(
        {
            **serif_rc(),
            "font.size": 9.0,
            "text.color": INK,
            "axes.edgecolor": "#A6B1BE",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": "#46566A",
            "ytick.color": INK,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.6,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.35, 5.25), facecolor="white")
    fig.text(0.034, 0.921, "Pre-release vs. post-release CRPS gain (1 bar)", va="top",
             fontsize=13.1, fontweight="bold", color="black")  # fmt: skip
    layouts = (
        ("(a) Other checkpoints", MODELS, 2.15, (-2, -1, 0, 1, 2), 0.862, 0.342, 0.451),
        ("(b) Kronos checkpoints", KRONOS, 12.0, (-10, -5, 0, 5, 10), 0.273, 0.120, 0.090),
    )
    left, right, gap = 0.167, 0.976, 0.025
    width = (right - left - 4 * gap) / 5
    for label, models, limit, ticks, title_y, bottom, height in layouts:
        fig.text(0.034, title_y, label, va="top", fontsize=10.2, fontweight="bold")
        for column, market in enumerate(MARKETS):
            ax = fig.add_axes((left + column * (width + gap), bottom, width, height))
            plot_market(ax, estimates, market, models, column == 0, (limit, ticks))
    fig.text((left + right) / 2, 0.046,
             "DiD in CRPS gain (target units \N{MULTIPLICATION SIGN} 100)",
             ha="center", va="bottom", fontsize=9.0)  # fmt: skip
    for extension in ("png", "pdf"):
        # cropped close to the content, so LaTeX sets the space around the figure; no dates
        metadata = {"CreationDate": None, "Creator": None} if extension == "pdf" else None
        fig.savefig(output.with_suffix(f".{extension}"), dpi=320, bbox_inches="tight",
                    pad_inches=0.01, metadata=metadata)  # fmt: skip
    plt.close(fig)


def main() -> None:
    """Draw fig:rq4 for the paper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=VERDICT)
    parser.add_argument("--output", type=Path, default=FIGURE,
                        help="Output base name for PNG and PDF")  # fmt: skip
    args = parser.parse_args()
    draw(load_estimates(args.csv), args.output)


if __name__ == "__main__":
    main()
