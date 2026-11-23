"""Analyse a run of 11-closures-length: the answer of Q1 and the results README.

Usage: .venv/bin/python experiments/11-closures-length/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import read_length_settings
from src.benchmark import SCORED
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("11-closures-length")


def _answer(folder: Path, run: str) -> Answer:
    # Q1: how many on-time models do better with the longer context (Holm, one-sided)
    settings = read_length_settings(load_study(folder / "closures-config.toml").options)
    tests = pd.read_csv(folder / "tables" / "length.csv")
    on_time = tests[(tests["family"] == "on time") & (tests["status"] == SCORED)]
    better = on_time[on_time["adjusted_p"] < settings.alpha]
    worse = on_time[on_time["skill_high"] < 0]
    lines = "; ".join(
        f"{model} {100 * skill:.1f} % [{100 * low:.1f}, {100 * high:.1f}], adjusted p {p:.2g}"
        for model, skill, low, high, p in on_time[
            ["model", "skill", "skill_low", "skill_high", "adjusted_p"]
        ].to_numpy()
    )
    late = tests[tests["family"] == "late"]
    late_better = int((late["adjusted_p"] < settings.alpha).sum())
    answer = "yes" if len(better) else "no"
    evidence = (
        f"Closed markets, test quarters, skill of the last 512 own bars (the benchmark input) "
        f"over the own bars of the last 512 calendar hours (trading hours only), both real "
        f"closes only: {len(better)} of {len(on_time)} on-time models are significantly better "
        f"with the longer context (one-sided Diebold-Mariano on closure days, Holm, "
        f"{settings.alpha}), {len(worse)} have the whole interval below zero: {lines}. Late "
        f"family: {late_better} of {len(late)} better. The encoding at the same bars "
        f"(forward-filled and missing against trading hours only) is 08-closures' Q1."
    )
    limitations = (
        "exploratory, planned after 08-closures' test results were seen; the benchmark "
        "input reaches further back in calendar time as well as holding more bars, so length "
        "and calendar reach are not separated; the benchmark input was descriptive and "
        "untested in 08's locked plan"
    )
    files = (f"{run}/tables/length.csv", f"{run}/tables/length_by_market.csv")
    return Answer("Q1", answer, evidence, limitations, files)


def main() -> None:
    """Answer Q1 and write the results README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    path = write_results_readme(invocation.study_dir, [_answer(folder, run)], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
