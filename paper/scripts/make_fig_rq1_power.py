"""Draw the paper's power figure (fig:power) from the committed 03-rq1-null results.

Reads ``tables/mde.csv`` (the MDE of every cell) and ``tables/mde_threshold.csv`` (the share of
cells with MDE at most X on 03's grid, the table 03's own figure ``figures/mde_threshold.pdf``
is drawn from) of the test run and the rule of 03's plan, keeps the testable cells exactly as
03 does (`src.benchmark.detectable_cells`), and draws them with
``src.figures.draw_power_summary``: (a) the share at the chosen X per horizon against the
required share, (b) the empirical distribution of the MDE per horizon. The drawing function
checks that the per-cell ECDF equals ``mde_threshold.csv`` at every grid value, so the figure
shows 03's numbers and no other. Run from the lab root:
``.venv/bin/python paper/scripts/make_fig_rq1_power.py``.
"""

from pathlib import Path

import pandas as pd

from src.benchmark import detectable_cells
from src.experiment import load_study
from src.figures import PowerLines, draw_power_summary

RUN = Path("experiments/03-rq1-null/results/test-2026-11-17")
"""The committed test results of 03-rq1-null."""

STUDY = Path("experiments/03-rq1-null/study.toml")
"""03's plan: the rule's threshold, required share and fewest winning quarters (the run writes
no config.toml of its own)."""

FIGURE = Path("paper/figures/fig_rq1_power.pdf")
"""The paper's figure."""

OUT = Path("paper/scripts/out")
"""The greyscale preview and the SVG copy (not in git); the 300 dpi PNG goes next to the PDF."""


def main() -> None:
    """Draw fig:power for the paper."""
    options = load_study(STUDY).options
    tables = RUN / "tables"
    # every scored cell, marked testable when it has the test quarters the rule needs (03's
    # detectable_cells); the untestable ones are drawn as their own outcome, not dropped
    mde = pd.read_csv(tables / "mde.csv")
    testable = detectable_cells(mde, min_winning_quarters=int(str(options["min_winning_quarters"])))
    cells = mde.assign(is_testable=mde.index.isin(testable.index))
    grid = [float(str(value)) for value in options["mde_threshold_grid_pct"]]  # type: ignore[union-attr]
    draw_power_summary(
        cells,
        pd.read_csv(tables / "mde_threshold.csv"),
        lines=PowerLines(
            chosen_pct=float(str(options["mde_threshold_pct"])),
            required_share=float(str(options["power_share"])),
            x_max=max(grid),
            power=float(str(options["power_target"])),
        ),
        pdf_path=FIGURE,
        grey_path=OUT / "fig_rq1_power_grey.png",
        exports=[OUT / "fig_rq1_power.svg", FIGURE.with_suffix(".png")],
    )


if __name__ == "__main__":
    main()
