"""Analyse a run of 12-horizon-spans and write the results README.

Usage: .venv/bin/python experiments/12-horizon-spans/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from src.experiment import Answer, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("12-horizon-spans")


def main() -> None:
    """Describe the spans of the longest horizon per market and write the README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    spans = pd.read_csv(folder / "tables" / "spans.csv")
    horizon = int(spans["horizon"].max())
    longest = spans[spans["horizon"] == horizon]
    parts = "; ".join(
        f"{market} median {median:.0f} h ({median / 24:.1f} days; quartiles {low:.0f} to "
        f"{high:.0f} h)"
        for market, median, low, high in zip(
            longest["market"].astype(str),
            longest["span_median_hours"].to_numpy(dtype=float),
            longest["span_q25_hours"].to_numpy(dtype=float),
            longest["span_q75_hours"].to_numpy(dtype=float),
            strict=True,
        )
    )
    evidence = (
        f"Test quarters, every labelled decision; a horizon of {horizon} own bars spans: "
        f"{parts}. Spans of every horizon are in spans.csv."
    )
    answer = Answer(
        "Q1",
        "yes",
        evidence,
        "descriptive; the span counts calendar hours, including the closed hours of nights, "
        "weekends and holidays, so a market's typical span depends on when its decisions fall",
        (f"{run}/tables/spans.csv",),
    )
    path = write_results_readme(invocation.study_dir, [answer], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
