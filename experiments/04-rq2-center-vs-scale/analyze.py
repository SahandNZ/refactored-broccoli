"""Analyse a run of 04-rq2-center-vs-scale and write the results README.

Usage: .venv/bin/python experiments/04-rq2-center-vs-scale/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import (
    CentreScaleRule,
    centre_scale_rule,
    centre_scale_verdict,
    figure_table,
)
from src.benchmark import (
    MODEL_NAMES,
    SCORED,
    paper_order,
    study_settings,
    unscored_note,
)
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme
from src.figures import RQ2_MARKETS, draw_centre_scale, read_centre_scale

logger = logging.getLogger("04-rq2-center-vs-scale")


def _skilled_names(skilled: pd.DataFrame) -> str:
    names = [
        f"{row.model} {row.market} {row.horizon}h (hit rate {row.hit_rate:.3f} "
        f"[{row.hit_low:.3f}, {row.hit_high:.3f}] against {row.reference_hit_rate:.3f}, "
        f"difference {row.hit_difference:.3f} [{row.diff_low:.3f}, {row.diff_high:.3f}], "
        f"adjusted one-sided p {row.adjusted_p:.2g}, rank IC {row.rank_ic:.3f})"
        for row in skilled.itertuples(index=False)
    ]
    return "; ".join(names) or "none"


def _centre_answer(
    verdict: pd.DataFrame, rule: CentreScaleRule, files: tuple[str, ...], limitations: str
) -> Answer:
    main = verdict[verdict["family"] == "main"]
    late = verdict[verdict["family"] == "late"]
    skilled = main[main["has_centre_skill"]]
    late_skilled = late[late["has_centre_skill"]]
    tested = int(main["p_value"].notna().sum())
    evidence = (
        f"{len(skilled)} of {len(main)} cells ({tested} with a centre test) of the main family "
        f"(models with at least "
        f"{rule.main_family_min_quarters} scored quarters wholly after release) have a median "
        f"that predicts the sign better than {rule.centre_reference}: {_skilled_names(skilled)}; "
        f"{int(main['has_worse_centre'].sum())} predict it worse (interval below 0). "
        f"Late family (its own Holm correction and verdict, "
        f"{'yes' if len(late_skilled) > 0 else 'no'}): {len(late_skilled)} of {len(late)} cells: "
        f"{_skilled_names(late_skilled)}." + unscored_note(verdict)
    )
    if tested == 0:
        evidence = (
            "No cell of the main family has a centre test (the family is empty, or every "
            "median is always zero or has fewer than 5 blocks of signed decision "
            "times), so Q1 "
            "cannot be answered. " + evidence
        )
        return Answer("Q1", "inconclusive", evidence, limitations, files)
    return Answer("Q1", "yes" if len(skilled) > 0 else "no", evidence, limitations, files)


def _scale_answer(
    verdict: pd.DataFrame, reference: str, files: tuple[str, ...], limitations: str
) -> Answer:
    calibrated = int(verdict["is_calibrated"].sum())
    better = int(verdict["beats_reference_scale"].sum())
    both = verdict[verdict["is_calibrated"] & verdict["beats_reference_scale"]]
    names = [
        f"{model} {market} {horizon}h (coverage {coverage:.3f} [{low:.3f}, {high:.3f}], "
        f"coverage error {mine:.3f} vs {theirs:.3f})"
        for model, market, horizon, coverage, low, high, mine, theirs in zip(
            both["model"],
            both["market"],
            both["horizon"],
            both["coverage"],
            both["coverage_low"],
            both["coverage_high"],
            both["coverage_error"],
            both["reference_coverage_error"],
            strict=True,
        )
    ]
    evidence = (
        f"{calibrated} of {len(verdict)} cells are calibrated and {better} track risk better "
        f"than {reference}; both: "
        + ("; ".join(names) or "none")
        + ". "
        + _coverage_range(verdict)
        + unscored_note(verdict)
    )
    return Answer("Q2", "yes" if names else "no", evidence, limitations, files)


def _coverage_range(verdict: pd.DataFrame) -> str:
    # the narrowest and the widest model: the mean 10-90 % coverage of its scored cells
    scored = verdict[verdict["status"] == SCORED]
    if scored.empty:
        return ""
    means = scored.groupby("model")["coverage"].mean().sort_values()
    return (
        f"Mean coverage of the 10-90 % interval per model (scored cells) runs from "
        f"{means.iloc[0]:.3f} ({means.index[0]}) to {means.iloc[-1]:.3f} ({means.index[-1]})."
    )


def main() -> None:
    """Apply the decision rule to every zero-shot cell, answer Q1 and Q2 and draw fig:rq2."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    rule = centre_scale_rule(config)
    alpha = study_settings(config).alpha
    cells = pd.read_csv(folder / "tables" / "cells.csv")
    quarters = pd.read_csv(folder / "tables" / "quarters.csv")
    verdict = centre_scale_verdict(
        cells[cells["kind"] == "zero-shot"],
        quarters[quarters["kind"] == "zero-shot"],
        alpha=alpha,
        rule=rule,
    )
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    # the figure also shows the scale reference, judged by the same rule on its own
    reference = centre_scale_verdict(
        cells[cells["model"] == rule.scale_reference],
        quarters[quarters["model"] == rule.scale_reference],
        alpha=alpha,
        rule=rule,
    )
    table = figure_table(pd.concat([reference, verdict]), horizon=rule.figure_horizon)
    table.to_csv(folder / "tables" / "rq2.csv", index=False)
    # the rows in tab:rq1's order (main family, then late, each in release order), paper names;
    # for the paper's models this is the paper figure's RQ2_MODELS
    families = dict(zip(table["model"].astype(str), table["family"].astype(str), strict=True))
    models = tuple((name, MODEL_NAMES.get(name, name)) for name in paper_order(families))
    present = set(table["market"].astype(str))
    markets = tuple(style for style in RQ2_MARKETS if style[0] in present)
    rq2_path = folder / "tables" / "rq2.csv"
    draw_centre_scale(
        read_centre_scale(rq2_path, models=models, markets=markets),
        pdf_path=folder / "figures" / "rq2.pdf",
        png_path=folder / "figures" / "rq2.png",
        models=models,
        markets=markets,
    )
    development = str(config.options["development_run"])
    limitations = (
        f"{config.folds} folds; only decisions after each model's release, so a model released "
        "inside the window has fewer quarters (late models form their own Holm family and "
        f"may not meet the {rule.min_quarters}-quarter condition); one-sided tests at "
        f"{alpha:g}, intervals unadjusted; the PIT is seen only at the nine forecast "
        "quantiles; the rank IC is pooled over the cell's assets and times; decisions where "
        "climatology's median is exactly zero (assets with many zero training returns) leave "
        "the centre test, so it rests on fewer decisions of those markets (signed_decisions; "
        "owner's decision, 2026-11-17); the reflection check (1/P) is not run; the run on the "
        f"development folds is {development} (results/{development}/ and the git history of "
        "this README)"
    )
    files = (
        f"{run}/tables/verdict.csv",
        f"{run}/tables/quarters.csv",
        f"{run}/tables/rq2.csv",
        f"{run}/figures/rq2.pdf",
    )
    answers = [
        _centre_answer(verdict, rule, files, limitations),
        _scale_answer(verdict, rule.scale_reference, files, limitations),
    ]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
