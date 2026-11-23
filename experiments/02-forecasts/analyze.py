"""Analyse a run of 02-forecasts: the validity table (Q1) and the results README.

Usage: .venv/bin/python experiments/02-forecasts/analyze.py --run-id ID
"""

import logging
from pathlib import Path

from lib import validity_table
from src.benchmark import cache_settings, read_cache
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("02-forecasts")


def main() -> None:
    """Count the valid forecasts of every model and answer Q1 of the plan, per window."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    table = validity_table(read_cache(folder), config)
    table.to_csv(folder / "tables" / "validity.csv", index=False)
    minimum = cache_settings(config).q1_min_share
    # only decisions after each model's release count; a model with none in a window has no
    # row there (or a NaN share) and does not decide that window
    scored = table[table["decisions"] > 0]
    below = scored[scored["share"] < minimum]
    totals = scored[scored["window"] == "all"]
    lowest = totals.sort_values("share").iloc[0]
    windows = [name for name in scored["window"].unique() if name != "all"]
    failed = sorted(set(below["window"]) - {"all"})
    evidence = (
        f"{len(totals)} models with decisions after their release; lowest share of labelled "
        f"decisions with a valid forecast {lowest['share']:.4f} ({lowest['model']}); per "
        f"window: " + ", ".join(f"{name} {'no' if name in failed else 'yes'}" for name in windows)
    )
    if not below.empty:
        evidence += f"; below {minimum:.0%}: " + ", ".join(below["model"] + " " + below["window"])
    exceptions = table[(table["not_fitted"] > 0) & (table["window"] != "all")]
    if not exceptions.empty:
        evidence += (
            "; listed exceptions (baseline fits that did not converge, not_fitted.csv): "
            + ", ".join(
                f"{model} {window} ({count} decisions)"
                for model, window, count in zip(
                    exceptions["model"], exceptions["window"], exceptions["not_fitted"], strict=True
                )
            )
        )
    limitations = (
        f"{config.folds} folds only; only decisions after each model's release count; valid "
        "means finite, ordered quantiles, not accurate"
    )
    answer = "no" if failed else "yes"
    answers = [Answer("Q1", answer, evidence + ".", limitations, (f"{run}/tables/validity.csv",))]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
