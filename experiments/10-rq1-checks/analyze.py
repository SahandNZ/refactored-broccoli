"""Analyse a run of 10-rq1-checks: the verdict of Q1 and the results README.

Usage: .venv/bin/python experiments/10-rq1-checks/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import SCORED, beats_baseline, null_rule
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme
from src.stats import holm

logger = logging.getLogger("10-rq1-checks")


def _verdict(cells: pd.DataFrame, *, alpha: float, min_winning_quarters: int) -> pd.DataFrame:
    # rq1's rule over one Holm family: every tested cell of the run's zero-shot models
    adjusted = holm(cells["p_value"].to_numpy(dtype=float))
    beats = beats_baseline(
        adjusted,
        cells["winning_quarters"].to_numpy(),
        cells["dm_statistic"].to_numpy(),
        alpha=alpha,
        min_winning_quarters=min_winning_quarters,
    )
    return cells.assign(adjusted_p=adjusted, beats_baseline=beats)


def _gap_sentence(gaps: pd.DataFrame, horizon: int) -> str:
    # the gain of the hourly models over Kronos on the same decision points, with intervals
    with_interval = gaps[gaps["gain_low"].notna()]
    above = int((with_interval["gain_low"] > 0).sum())
    return (
        f"On the same decision points at {horizon} bar(s), the hourly models' gain over Kronos is "
        f"{gaps['gain'].min():.1f} % to {gaps['gain'].max():.1f} % (median "
        f"{np.median(gaps['gain']):.1f} %); {above} of {len(with_interval)} pairs of hourly "
        f"model, Kronos checkpoint and market have the whole 95 % interval above zero."
    )


def _reopening_sentence(reopening: pd.DataFrame, closure_hours: float) -> str:
    # the shares of decisions at a reopening, Kronos's against all, on the markets that close
    closing = reopening[reopening["market"] != "crypto"]
    shares = "; ".join(
        f"{market} {100 * grid:.1f} % of Kronos's decisions against {100 * share:.1f} % of all"
        for market, grid, share in zip(
            closing["market"],
            closing["grid_reopening_share"],
            closing["reopening_share"],
            strict=True,
        )
    )
    across = "; ".join(
        f"{market} {100 * grid:.1f} % against {100 * share:.1f} %"
        for market, grid, share in zip(
            closing["market"],
            closing["grid_target_closure_share"],
            closing["target_closure_share"],
            strict=True,
        )
    )
    return (
        f"Made at the close of the first bar after a closure of at least {closure_hours:g} hours "
        f"(the gap in the last input bar): {shares}. Target across such a closure (made at the "
        f"last close before it): {across}."
    )


def _q1(folder: Path, run: str) -> Answer:
    # Q1: the cells on Kronos's decision points, the cells that pass, the gaps and reopenings
    rq1 = load_study(folder / "rq1-config.toml")
    config = load_study(folder / "config.toml")
    rule = null_rule(rq1)
    alpha = float(str(rq1.options["alpha"]))
    cells = pd.read_csv(folder / "tables" / "cells.csv")
    verdict = _verdict(cells, alpha=alpha, min_winning_quarters=rule.min_winning_quarters)
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    tested = verdict[verdict["status"] == SCORED]
    winners = tested[tested["beats_baseline"]]
    names = ", ".join(
        f"{m} {k} {h}" for m, k, h in winners[["model", "market", "horizon"]].to_numpy()
    )
    evidence = (
        f"Every zero-shot model of rq1's main family on the decisions of "
        f"{config.options['grid_model']} and its grid, test quarters t1-t4, rq1's rule, one Holm "
        f"family of {len(tested)} tested cells: {len(winners)} cells beat the best baseline"
        f"{': ' + names if names else ''}. "
        + _gap_sentence(
            pd.read_csv(folder / "tables" / "kronos_gaps.csv"),
            int(str(config.options["gap_horizon"])),
        )
        + " "
        + _reopening_sentence(
            pd.read_csv(folder / "tables" / "reopening.csv"),
            float(str(config.options["min_closure_hours"])),
        )
    )
    times = tested["times"].astype(int)
    limitations = (
        "exploratory, planned after rq1's test results were seen; "
        f"{times.min()} to {times.max()} decision points per cell, so the tests have little "
        "power; the best baseline is chosen on "
        "all development decisions, as in rq1, not on Kronos's points; the gain over Kronos "
        "has an interval but no multiple-testing correction"
    )
    files = tuple(f"{run}/tables/{name}.csv" for name in ("verdict", "kronos_gaps", "reopening"))
    return Answer("Q1", "yes" if len(winners) else "no", evidence, limitations, files)


def main() -> None:
    """Apply rq1's rule on Kronos's decision points and write the results README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    answers = [_q1(folder, run)]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
