"""Analyse a run of 03-rq1-null and write the results README.

Usage: .venv/bin/python experiments/03-rq1-null/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import null_note, null_verdict, size_note, whole_quarter_note
from src.benchmark import (
    NullRule,
    detectable_cells,
    null_rule,
    study_settings,
    threshold_shares,
    unscored_note,
)
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme
from src.figures import draw_threshold_shares

logger = logging.getLogger("03-rq1-null")

SKILL_COLUMNS = [
    "model", "family", "market", "horizon", "status", "baseline", "times", "skill",
    "skill_low", "skill_high", "adjusted_p", "beats_baseline",
]  # fmt: skip
"""The columns of ``tables/skill.csv``, the table the paper's skill table is drawn from."""


def _family_answer(verdict: pd.DataFrame, family: str) -> tuple[str, str]:
    # the cells of one Holm family that beat the best baseline, and their sentence; a family
    # without a scored cell has no answer (inconclusive)
    cells = verdict[verdict["family"] == family]
    winners = cells[cells["beats_baseline"]]
    names = [
        f"{model} {market} {horizon} bars (skill {skill:.2f} %, adjusted p {p:.2g})"
        for model, market, horizon, skill, p in zip(
            winners["model"],
            winners["market"],
            winners["horizon"],
            winners["skill"],
            winners["adjusted_p"],
            strict=True,
        )
    ]
    scored = int((cells["status"] == "scored").sum())
    models = ", ".join(sorted(set(cells["model"]))) or "none"
    sentence = (
        f"{family} family ({models}): {len(winners)} of {scored} scored cells beat the best "
        "baseline: " + ("; ".join(names) or "none")
    )
    if scored == 0:
        return "inconclusive", sentence + " (no scored cell, so no test)"
    return ("yes" if names else "no"), sentence


def _late_mde(mde: pd.DataFrame, rule: NullRule) -> str:
    # each late model's MDE at the judged horizon: median over its scored markets; a model
    # with fewer quarters than the rule needs can never be detected
    late = mde[(mde["family"] == "late") & (mde["horizon"] == rule.mde_horizon)]
    parts: list[str] = []
    for model, rows in late.groupby("model"):
        if (rows["quarters"] < rule.min_winning_quarters).all():
            quarters = int(rows["quarters"].max())
            parts.append(f"{model}: {quarters} quarter(s), the rule cannot detect it")
            continue
        reached = rows[rows["is_reached"]]
        median = f"{reached['mde_pct'].median():.2f} %" if len(reached) else "never reached"
        power = f"{rule.power_target * 100:g} % power"
        parts.append(f"{model} {median} ({len(reached)} of {len(rows)} markets reach {power})")
    return "; ".join(parts) or "no late model has a scored cell"


def _power_answer(
    mde: pd.DataFrame,
    rule: NullRule,
    *,
    verdict: pd.DataFrame,
    resamples: int,
    folder: Path,
    run: str,
) -> Answer:
    # Q2: the share of the scored cells at the judged horizon that the rule can detect (enough
    # quarters) whose MDE is at or below the threshold X, against the rule's share K (and the
    # sensitivity shares)
    detectable = detectable_cells(mde, min_winning_quarters=rule.min_winning_quarters)
    shares = threshold_shares(detectable, thresholds=list(rule.threshold_grid_pct))
    shares.to_csv(folder / "tables" / "mde_threshold.csv", index=False)
    figures = folder / "figures"
    draw_threshold_shares(
        shares,
        chosen_pct=rule.mde_threshold_pct,
        pdf_path=figures / "mde_threshold.pdf",
        grey_path=figures / "mde_threshold_grey.png",
    )
    cells = detectable[detectable["horizon"] == rule.mde_horizon]
    left_out = int((mde["horizon"] == rule.mde_horizon).sum()) - len(cells)
    is_resolved = cells["is_reached"] & (cells["mde_pct"] <= rule.mde_threshold_pct)
    share = float(is_resolved.mean())
    rows = [
        {"share_needed": needed, "share": share, "is_powered": share >= needed, "is_main": main}
        for needed, main in [(rule.power_share, True)]
        + [(other, False) for other in rule.power_share_sensitivity]
    ]
    pd.DataFrame(rows).to_csv(folder / "tables" / "power_rule.csv", index=False)
    sensitivity = "; ".join(
        f"K = {row['share_needed']:.0%}: {'yes' if row['is_powered'] else 'no'}" for row in rows[1:]
    )
    evidence = (
        f"{int(is_resolved.sum())} of the {len(cells)} scored cells at {rule.mde_horizon} bar(s) "
        f"with at least {rule.min_winning_quarters} quarters "
        f"({share:.0%}) have an MDE at or below X = {rule.mde_threshold_pct:g} % of the best "
        f"baseline's CRPS, against K = {rule.power_share:.0%} (sensitivity, not deciding: "
        f"{sensitivity}); {left_out} cells with fewer quarters, which the rule cannot detect, "
        "are left out."
        + whole_quarter_note(cells, verdict, rule, share=share)
        + null_note(mde, verdict, resamples=resamples)
    )
    limitations = (
        "the planted edge is a constant CRPS gain added to the real model's centred loss "
        "difference, so it has the real noise and autocorrelation but not the shape of a real "
        "edge; Holm is replaced by its threshold for the most significant cell of each family "
        "(conservative); X comes from 09-mde-threshold"
    )
    files = (
        f"{run}/tables/mde_threshold.csv",
        f"{run}/figures/mde_threshold.pdf",
        f"{run}/tables/power_rule.csv",
        f"{run}/tables/mde.csv",
        f"{run}/tables/power.csv",
    )
    answer = "yes" if share >= rule.power_share else "no"
    return Answer("Q2", answer, evidence, limitations, files)


def main() -> None:
    """Apply the decision rules to every cell and answer Q1 and Q2."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    settings, rule = study_settings(config), null_rule(config)
    verdict = null_verdict(
        pd.read_csv(folder / "tables" / "cells.csv"),
        alpha=settings.alpha,
        min_winning_quarters=rule.min_winning_quarters,
    )
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    verdict[SKILL_COLUMNS].to_csv(folder / "tables" / "skill.csv", index=False)
    mde = pd.read_csv(folder / "tables" / "mde.csv")
    main_answer, main_sentence = _family_answer(verdict, "main")
    _, late_sentence = _family_answer(verdict, "late")
    evidence = (
        f"Main verdict, {main_sentence}. Separate verdict, {late_sentence}; the late models' "
        f"MDE at {rule.mde_horizon} bar(s): {_late_mde(mde, rule)}." + unscored_note(verdict)
    )
    at_horizon = mde[mde["horizon"] == rule.mde_horizon]
    limitations = (
        f"{config.folds} folds; only decisions after each model's scoring start, so a model "
        "released inside the window has fewer quarters (late family); a 'no' rules out only "
        f"edges the rule can detect: at {rule.mde_horizon} bar(s) its MDE "
        f"({rule.power_target * 100:g} % power) is {at_horizon['mde_pct'].median():.2f} % of "
        f"the baseline's CRPS in the median cell that reaches it, and "
        f"{int((~at_horizon['is_reached']).sum())} cells never reach it (Q2); a late model "
        f"with fewer than {rule.min_winning_quarters} quarters can never pass the quarter "
        "condition; the cells at 24 bars, whose labels overlap most, may reject more often "
        f"than the nominal level ({size_note(invocation.study_dir)})"
    )
    files = (
        f"{run}/tables/verdict.csv",
        f"{run}/tables/skill.csv",
        f"{run}/tables/quarters.csv",
    )
    answers = [
        Answer("Q1", main_answer, evidence, limitations, files),
        _power_answer(
            mde,
            rule,
            verdict=verdict,
            resamples=settings.bootstrap.n_resamples,
            folder=folder,
            run=run,
        ),
    ]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
