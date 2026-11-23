"""Analyse a run of 13-context-baselines: the verdict of Q1 and the results README.

Usage: .venv/bin/python experiments/13-context-baselines/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import SCORED, beats_baseline, null_rule
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme
from src.stats import holm

logger = logging.getLogger("13-context-baselines")

RQ1_VERDICT = Path("03-rq1-null") / "results" / "test-2026-11-17" / "tables" / "verdict.csv"
"""rq1's committed test verdict, relative to the paper folder (skill against rq1's baselines)."""


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


def _compared(verdict: pd.DataFrame, rq1: pd.DataFrame) -> pd.DataFrame:
    # each cell's skill against the context baseline next to its skill in rq1, at 1 bar
    keys = ["model", "market", "horizon"]
    ours = verdict.loc[verdict["status"] == SCORED, [*keys, "baseline", "skill"]]
    theirs = rq1.loc[rq1["horizon"] == 1, [*keys, "baseline", "skill"]]
    return ours.merge(theirs, on=keys, suffixes=("", "_rq1"), validate="one_to_one")


def main() -> None:
    """Apply rq1's rule against the context baselines and write the results README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    rq1 = load_study(folder / "rq1-config.toml")
    alpha = float(str(rq1.options["alpha"]))
    cells = pd.read_csv(folder / "tables" / "cells.csv")
    verdict = _verdict(cells, alpha=alpha, min_winning_quarters=null_rule(rq1).min_winning_quarters)
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    compared = _compared(verdict, pd.read_csv(invocation.study_dir.parent / RQ1_VERDICT))
    compared.to_csv(folder / "tables" / "compared.csv", index=False)
    failed = pd.read_csv(folder / "tables" / "garch_not_fitted.csv")
    tested = verdict[verdict["status"] == SCORED]
    winners = tested[tested["beats_baseline"]]
    chosen = compared.drop_duplicates("market")
    baselines = ", ".join(
        f"{market} {name}"
        for market, name in zip(chosen["market"], chosen["baseline"], strict=True)
    )
    evidence = (
        f"Test quarters, 1 bar, rq1's rule with the random walk, climatology, GARCH(1,1) and "
        f"the ridge regression refitted on the last 512 own bars: {len(winners)} of "
        f"{len(tested)} tested cells beat the best baseline. Best baseline per market: "
        f"{baselines}. Skill against it: median {np.median(compared['skill']):.1f} % "
        f"(rq1: {np.median(compared['skill_rq1']):.1f} %); {int((compared['skill'] > 0).sum())} "
        f"positive (rq1: {int((compared['skill_rq1'] > 0).sum())}). GARCH daily fits that did "
        f"not converge: {int(failed['failed'].sum())} (their days have no GARCH forecast)."
    )
    limitations = (
        "exploratory, planned after rq1's test results were seen; one horizon (1 bar); a "
        "decision needs 512 earlier decisions of the cache, so the first months of d1 have no "
        "context baseline on the markets that close and the best baseline is chosen on fewer "
        "development decisions; GARCH is refitted daily, not at every decision"
    )
    files = tuple(
        f"{run}/tables/{name}.csv" for name in ("verdict", "compared", "garch_not_fitted")
    )
    answer = Answer("Q1", "yes" if len(winners) else "no", evidence, limitations, files)
    path = write_results_readme(invocation.study_dir, [answer], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
