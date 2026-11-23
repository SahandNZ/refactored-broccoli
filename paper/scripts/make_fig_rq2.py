"""Draw the paper's RQ2 figure (fig:rq2) from the committed 04-rq2-center-vs-scale results.

Reads ``tables/rq2.csv`` of the test run (written by 04's ``analyze.py``) and draws it with
``src.figures.draw_centre_scale`` (the owner's layout: one panel per market). Writes
``paper/figures/fig_rq2.pdf`` and ``paper/figures/fig_rq2.png``.
Run from the lab root: ``.venv/bin/python paper/scripts/make_fig_rq2.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.figures import draw_centre_scale, read_centre_scale

SOURCE = Path("experiments/04-rq2-center-vs-scale/results/test-2026-11-17/tables/rq2.csv")
"""The committed test-run table of 04-rq2-center-vs-scale."""

FIGURE = Path("paper/figures/fig_rq2.pdf")
"""The paper's figure."""

PREVIEW = Path("paper/figures/fig_rq2.png")
"""The PNG copy, next to the PDF."""


def main() -> None:
    """Draw fig:rq2 for the paper."""
    draw_centre_scale(read_centre_scale(SOURCE), pdf_path=FIGURE, png_path=PREVIEW)


if __name__ == "__main__":
    main()
