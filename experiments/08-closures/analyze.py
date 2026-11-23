"""Analyse a run of 08-closures: test the representations and write the results README.

The body of the paper's table comes from ``make_tab_rq5_closures.py`` on the same run.

Usage: .venv/bin/python experiments/08-closures/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import (
    ALL,
    BASELINE,
    CLOSED,
    CONTROL,
    LATE,
    ON_TIME,
    REFERENCE,
    REPRESENTATIONS,
    ClosureSettings,
    analysis_settings,
    best_representations,
    closure_tests,
    closure_verdict,
    control_answer,
    family_answer,
    sorted_robustness,
)
from src.benchmark import SCORED
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("08-closures")


def _test_text(alternative: str, row: pd.Series) -> str:
    # one test in words: its skill and interval in % and its adjusted p, or why not tested
    if row["status"] != SCORED:
        return f"{alternative}: {row['status']}"
    skill, low, high = 100 * row["skill"], 100 * row["skill_low"], 100 * row["skill_high"]
    worse = ", worse (whole interval below zero)" if row["is_worse"] else ""
    return (
        f"{alternative}: skill {skill:.1f} % [{low:.1f}, {high:.1f}], "
        f"adjusted one-sided p {row['adjusted_p']:.2g}{worse}"
    )


def _family_evidence(verdict: pd.DataFrame, family: str) -> tuple[str, str]:
    # one Holm family's answer (`family_answer`) and its evidence, one line per model
    answer = family_answer(verdict, family)
    lines: list[str] = []
    for model, rows in verdict.loc[verdict["family"] == family].groupby("model", sort=False):
        tests = "; ".join(
            _test_text(str(alternative), row)
            for alternative, (_, row) in zip(rows["alternative"], rows.iterrows(), strict=True)
        )
        days = int(rows["days"].iloc[0])
        lines.append(f"{model} {rows['verdict'].iloc[0]} ({days} closure days; {tests})")
    tested = verdict.loc[(verdict["family"] == family) & (verdict["status"] == SCORED), "model"]
    winners = verdict.loc[verdict["model"].isin(tested) & (verdict["verdict"] != REFERENCE)]
    evidence = (
        f"{family} family: {answer}, {winners['model'].nunique()} of {len(lines)} models "
        f"({tested.nunique()} tested) feed better with an alternative than with {REFERENCE}: "
        + ("; ".join(lines) or "none")
    )
    return answer, evidence


def _invalid_counts(summary: pd.DataFrame, group: str) -> str:
    # the invalid forecasts of each zero-shot model's representations in one group
    pooled = summary.loc[
        (summary["split"] == ALL)
        & (summary["group"] == group)
        & summary["representation"].isin(REPRESENTATIONS)
        & (summary["model"] != BASELINE)
    ]
    invalid = pooled.loc[pooled["invalid"].fillna(0) > 0]
    counts = [
        f"{model} {arm} {int(count)}"
        for model, arm, count in zip(
            invalid["model"], invalid["representation"], invalid["invalid"], strict=True
        )
    ]
    return "; ".join(counts) or "none"


def _short_contexts(summary: pd.DataFrame, min_own_closes: int) -> str:
    # the decisions each group dropped for a too short trading-hours context (a data gap)
    counts = summary.groupby("group", sort=False)["short_contexts"].max()
    groups = "; ".join(f"{group}: {int(count)}" for group, count in counts.items())
    return (
        f"Decisions dropped from every arm for fewer than {min_own_closes} own closes in the "
        f"trading-hours context: {groups or 'none'}."
    )


def _percent(value: float) -> str:
    # a skill in %, or why there is none
    return "not scored" if pd.isna(value) else f"{100 * value:.1f} %"


def _robustness_text(robustness: pd.DataFrame | None) -> str:
    # the skills with sorted quantiles next to the primary ones (descriptive, untested)
    if robustness is None:
        return "Sorted-quantile robustness: not computed in this run (no crps_sorted)."
    pairs = "; ".join(
        f"{family} {model} {alternative} {_percent(sorted_skill)} (primary {_percent(skill)}, "
        f"{int(decisions_sorted)} vs {int(decisions)} decisions)"
        for family, model, alternative, sorted_skill, skill, decisions_sorted, decisions in zip(
            robustness["family"], robustness["model"], robustness["alternative"],
            robustness["skill_sorted"], robustness["skill"], robustness["decisions_sorted"],
            robustness["decisions"], strict=True,
        )
    )  # fmt: skip
    return (
        "Robustness, descriptive and untested: skill with the nine quantiles sorted "
        f"(monotone rearrangement, robustness.csv): {pairs or 'none'}."
    )


def _representation_answer(
    verdict: pd.DataFrame,
    summary: pd.DataFrame,
    robustness: pd.DataFrame | None,
    settings: ClosureSettings,
    run: str,
) -> Answer:
    # Q1: the family of models on time decides; the late family has its own verdict, reported
    answer, on_time = _family_evidence(verdict, ON_TIME)
    late_answer, late = _family_evidence(verdict, LATE)
    not_scored = summary.loc[(summary["split"] == ALL) & (summary["status"] != SCORED)]
    untested = [
        f"{model} / {group} / {arm}: {status}"
        for model, group, arm, status in zip(
            not_scored["model"], not_scored["group"], not_scored["representation"],
            not_scored["status"], strict=True,
        )
    ]  # fmt: skip
    counts = _invalid_counts(summary, CLOSED)
    evidence = (
        f"{on_time}. Separate verdict, {late_answer}: {late}. "
        f"Arms not scored (model / group / arm, summary.csv): {'; '.join(untested) or 'none'}. "
        "Invalid forecasts on the closed markets, which remove the decision from every "
        f"representation of the model: {counts}. "
        + _short_contexts(summary, settings.min_own_closes)
        + " "
        + _robustness_text(robustness)
    )
    small_sample_days = settings.small_sample_days
    limitations = (
        "one-step forecasts of the first bars after a closure only; every representation sees "
        "the same calendar hours, so trading hours only holds fewer values than the "
        "benchmark's input (reported as a descriptive arm); an exchange that opens off the "
        "hour grid (US and Hong Kong stocks, 09:30 local) reopens with a half bar; decisions "
        "only after each model's release; Toto 2.0's shorter trading-hours contexts are "
        "left-padded with their first value to a multiple of its patch (32); Kronos and the "
        "joint-input arm are not coded; with dependent closure days the one-sided test is "
        "somewhat liberal (measured at alpha 0.05: about 6.5 % with 250 days, README Plan "
        "changes)"
    )
    tested_days = verdict.loc[verdict["status"] == SCORED, "days"]
    if len(tested_days) and tested_days.min() < small_sample_days:
        limitations += (
            f"; a model tested on fewer than {small_sample_days} closure days (here down to "
            f"{int(tested_days.min())}) has a one-sided test that rejects about 8 % of the "
            "time, measured at alpha 0.05 with 40 dependent days, more with fewer (README, Plan "
            "changes)"
        )
    files = (f"{run}/tables/verdict.csv", f"{run}/tables/summary.csv")
    if robustness is not None:
        files = (*files, f"{run}/tables/robustness.csv")
    return Answer("Q1", answer, evidence, limitations, files)


def _control_answer(
    best: pd.DataFrame,
    summary: pd.DataFrame,
    control_assets: int,
    settings: ClosureSettings,
    run: str,
) -> Answer:
    # Q2: yes when every model has the same best representation on both groups
    reopen_bars, min_own_closes = settings.reopen_bars, settings.min_own_closes
    differ = best.loc[~best["is_same"]]
    lines = [
        f"{model} (closed markets {closed}; crypto {control})"
        for model, closed, control in zip(
            differ["model"], differ["closed_best"], differ["control_best"], strict=True
        )
    ]
    answer = control_answer(best)
    evidence = (
        f"{int(best['is_same'].sum())} of {len(best)} models have the same best representation "
        "on crypto with its weekends cut as on the closed markets; different: "
        + ("; ".join(lines) or "none")
        + ". Invalid forecasts on the control, which remove the decision from every "
        f"representation of the model: {_invalid_counts(summary, CONTROL)}. "
        + _short_contexts(summary.loc[summary["group"] == CONTROL], min_own_closes)
    )
    limitations = (
        "the best representation by mean CRPS without a test or an interval (an exemption "
        "from experiments rule 13, README Plan changes), so a best inside the noise "
        f"counts; the control is the {control_assets} crypto assets at one weekly time, the "
        f"48-hour weekend cut with decisions on Mondays 00:00-{reopen_bars - 1:02d}:00 UTC (the "
        "exchange schedule's time), "
        "while the real "
        "closures are also nights and holidays at many times of day; Toto 2.0's shorter "
        "trading-hours contexts are left-padded with their first value to a multiple of its "
        "patch (32)"
    )
    return Answer("Q2", answer, evidence, limitations, (f"{run}/tables/best.csv",))


def main() -> None:
    """Test the representations of every model (Q1), compare the control (Q2), write the README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    summary = pd.read_csv(folder / "tables" / "summary.csv")
    settings = analysis_settings(invocation.study_dir, config, summary)
    losses = pd.read_parquet(folder / "metrics.parquet")
    verdict = closure_verdict(closure_tests(losses, settings), alpha=settings.alpha)
    best = best_representations(summary)
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    best.to_csv(folder / "tables" / "best.csv", index=False)
    # a run written before the sorted scores existed has no robustness row
    robustness = None
    if "crps_sorted" in losses.columns:
        robustness = sorted_robustness(losses, verdict)
        robustness.to_csv(folder / "tables" / "robustness.csv", index=False)
    control_assets = losses.loc[losses["group"] == CONTROL, "asset"].nunique()
    answers = [
        _representation_answer(verdict, summary, robustness, settings, run),
        _control_answer(best, summary, control_assets, settings, run),
    ]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
