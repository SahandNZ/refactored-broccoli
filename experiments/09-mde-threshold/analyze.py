"""Analyse a run of 09-mde-threshold: choose rq1's MDE threshold and write the results README.

Usage: .venv/bin/python experiments/09-mde-threshold/analyze.py --run-id ID
"""

import logging
import math
from pathlib import Path

import pandas as pd

from src.benchmark import detectable_cells, null_rule, powered_threshold, threshold_shares
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme
from src.figures import draw_mde, draw_threshold_shares

logger = logging.getLogger("09-mde-threshold")


def main() -> None:
    """Sweep the threshold, take the smallest one that powers rq1's rule and answer Q1."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    rule = null_rule(load_study(folder / "rq1-config.toml"))
    mde = pd.read_csv(folder / "tables" / "mde.csv")
    grid = list(rule.threshold_grid_pct)
    # only the cells with enough quarters for the rq1 rule can ever be detected
    detectable = detectable_cells(mde, min_winning_quarters=rule.min_winning_quarters)
    shares = threshold_shares(detectable, thresholds=grid)
    shares.to_csv(folder / "tables" / "mde_threshold.csv", index=False)
    horizon, needed = rule.mde_horizon, rule.power_share
    chosen = powered_threshold(shares, share=needed, horizon=horizon)
    figures = folder / "figures"
    draw_threshold_shares(
        shares,
        chosen_pct=chosen,
        pdf_path=figures / "mde_threshold.pdf",
        grey_path=figures / "mde_threshold_grey.png",
    )
    draw_mde(
        mde, threshold_pct=chosen, pdf_path=figures / "mde.pdf", grey_path=figures / "mde_grey.png"
    )
    column = f"share_{horizon}h"
    cells = int((detectable["horizon"] == horizon).sum())
    left_out = int((mde["horizon"] == horizon).sum()) - cells
    if cells == 0:
        evidence = (
            f"no scored cell at {horizon} bar(s) has the {rule.min_winning_quarters} quarters the "
            "rq1 rule needs, so no threshold can be chosen."
        )
    elif math.isnan(chosen):
        evidence = (
            f"at most {shares[column].iloc[-1]:.0%} of the {cells} cells at {horizon} bar(s) "
            f"have an MDE at or below {grid[-1]:g} %, short of {needed:.0%}: no threshold of "
            "the grid powers the rule."
        )
    else:
        at_threshold = shares.loc[shares["threshold_pct"] == chosen, column].iloc[0]
        evidence = (
            f"threshold {chosen:g} % of the best baseline's CRPS, the smallest of the grid at "
            f"which at least {needed:.0%} of the {cells} cells at {horizon} bar(s) have an MDE "
            f"at or below it ({at_threshold:.0%}); set it as mde_threshold_pct of 03-rq1-null."
        )
    evidence += (
        f" {left_out} cells at {horizon} bar(s) with fewer than {rule.min_winning_quarters} "
        "quarters, which the rule cannot detect, are left out."
    )
    limitations = (
        "development folds d1-d4 only, which overlap some releases (each model is scored after "
        "its release, so late models have fewer decisions, and a model released after d4 has "
        "no cell); the planted edge is a constant CRPS gain on the real model's centred loss "
        "difference; MDEs between grid gains are interpolated linearly; the shared bootstrap "
        "tests reject somewhat more often than their nominal level with overlapping labels "
        "and few blocks (size check: tables/bootstrap_size.csv)"
    )
    files = (
        f"{run}/tables/mde_threshold.csv",
        f"{run}/figures/mde_threshold.pdf",
        f"{run}/tables/mde.csv",
        f"{run}/figures/mde.pdf",
        f"{run}/tables/power.csv",
        f"{run}/tables/bootstrap_size.csv",
    )
    answer = Answer("Q1", "no" if math.isnan(chosen) else "yes", evidence, limitations, files)
    path = write_results_readme(invocation.study_dir, [answer], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
