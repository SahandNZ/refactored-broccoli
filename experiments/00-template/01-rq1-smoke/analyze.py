"""Analyse a run of 01-rq1-smoke: the timing figure and the results README.

Usage: .venv/bin/python experiments/00-template/01-rq1-smoke/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from src.experiment import Answer, parse_invocation, study_runs, write_results_readme
from src.figures import draw_timing

logger = logging.getLogger("01-rq1-smoke")


def main() -> None:
    """Answer Q1 from the checks table and draw forecast time and GPU memory per model."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    checks = pd.read_csv(folder / "tables" / "checks.csv")
    timing = checks[["model", "predict_seconds", "peak_gpu_gb"]]
    timing.to_csv(folder / "tables" / "timing.csv", index=False)
    draw_timing(
        timing,
        pdf_path=folder / "figures" / "timing.pdf",
        grey_path=folder / "figures" / "timing_grey.png",
    )
    failed = checks.loc[~checks["passed"], "model"].tolist()
    share = checks["passed"].mean()
    answer = "yes" if not failed else "no"
    passed = int(checks["passed"].sum())
    evidence = f"{passed} of {len(checks)} models pass every check (share {share:.2f})"
    if failed:
        evidence += "; failed: " + ", ".join(failed)
    answers = [
        Answer(
            "Q1", answer, evidence + ".",
            "24 recent contexts of 3 assets only; a smoke check of the wrappers, not of accuracy",
            (f"{run}/tables/checks.csv", f"{run}/tables/timing.csv", f"{run}/figures/timing.pdf"),
        )
    ]  # fmt: skip
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
