"""Analyse a run of 06-rq4-leakage and write the results README.

Usage: .venv/bin/python experiments/06-rq4-leakage/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import (
    MARKET_LABELS,
    TESTS,
    figure_families,
    figure_limit,
    figure_table,
    leakage_options,
    leakage_verdict,
)
from src.benchmark import (
    MARKET_NAMES,
    MODEL_NAMES,
    SCORED,
    study_settings,
    unscored_note,
)
from src.experiment import (
    Answer,
    StudyConfig,
    load_study,
    parse_invocation,
    study_runs,
    write_results_readme,
)
from src.figures import draw_leakage

logger = logging.getLogger("06-rq4-leakage")

SIZE_HORIZON = 24
"""The horizon in bars whose overlapping labels make the test somewhat liberal (size check)."""

SIZE_CHECK_SHORTEST = 400
"""The shortest period (decision times) of the size check in the plan README."""


def _cells(rows: pd.DataFrame) -> str:
    # the cells of `rows` with their difference, interval and Holm-adjusted p
    return "; ".join(
        f"{model} {market} {horizon}h (did {did:.4f}, interval {low:.4f} .. {high:.4f}, "
        f"adjusted p {p:.2g})"
        for model, market, horizon, did, low, high, p in zip(
            rows["model"],
            rows["market"],
            rows["horizon"],
            rows["did"],
            rows["low"],
            rows["high"],
            rows["adjusted_p"],
            strict=True,
        )
    )


def _unscored_cells(primary: pd.DataFrame) -> str:
    # the cells of the test leakage that are not scored, by name (unscored_note groups them
    # by status only)
    rows = primary[primary["status"] != SCORED]
    if rows.empty:
        return ""
    cells = ", ".join(
        f"{model} {market} {horizon}h"
        for model, market, horizon in zip(
            rows["model"], rows["market"], rows["horizon"], strict=True
        )
    )
    return f" The cells not scored: {cells}."


def _checks(verdict: pd.DataFrame) -> str:
    # one sentence per reported check: how many of its scored cells reject after its own Holm,
    # and which
    sentences: list[str] = []
    for test in TESTS:
        family = verdict[(verdict["test"] == test) & (verdict["status"] == SCORED)]
        if test == "leakage" or family.empty:
            continue
        first, second = TESTS[test]
        rejected = family[family["rejects"]]
        listed = f": {_cells(rejected)}" if len(rejected) else ""
        sentences.append(
            f" Check {test} (skill {first} minus {second}, report only): Holm rejects "
            f"{len(rejected)} of {len(family)} scored cells{listed}."
        )
    return "".join(sentences)


def _size_note(primary: pd.DataFrame, hourly: set[str]) -> str:
    # the size limitation with the after-release lengths of this run's scored 24-bar cells of
    # the models that decide every hour (on the 4-hour grid their labels overlap; Kronos's
    # 167-hour grid does not), and whether the shortest is below the lengths the check covers
    long = primary[
        (primary["horizon"] == SIZE_HORIZON)
        & (primary["status"] == SCORED)
        & primary["model"].isin(hourly)
    ]
    lengths = ""
    if len(long):
        shortest, longest = int(long["second_times"].min()), int(long["second_times"].max())
        below = (
            f", the shortest below the {SIZE_CHECK_SHORTEST} the check covers"
            if shortest < SIZE_CHECK_SHORTEST
            else ""
        )
        lengths = (
            f" (this run, models that decide every hour: {shortest}-{longest} times after "
            f"the release{below})"
        )
    return (
        f"the test is somewhat liberal for {SIZE_HORIZON}-bar cells: in the synthetic size "
        f"check with overlapping labels it rejects 8.3-8.5 % at 5 % for periods of "
        f"{SIZE_CHECK_SHORTEST}-2,400 times and 5.3 % at 8,800{lengths}"
    )


def _left_out(listed: pd.DataFrame, primary: pd.DataFrame) -> str:
    # the limitation on baseline fits that did not converge (the run's not_fitted.csv) and
    # the cells whose control lost decisions (``unpaired`` of the test leakage)
    fits = ", ".join(
        f"{model} {window} {asset}"
        for model, window, asset in zip(
            listed["model"], listed["window"], listed["asset"], strict=True
        )
    )
    touched = primary[(primary["unpaired"] > 0) & (primary["status"] == SCORED)]
    per_cell = touched.groupby(["market", "horizon"], as_index=False)["unpaired"].max()
    cells = ", ".join(
        f"{market} {horizon}h (at most {count} decisions of one model)"
        for market, horizon, count in zip(
            per_cell["market"], per_cell["horizon"], per_cell["unpaired"], strict=True
        )
    )
    return (
        f"baseline fits that did not converge, left out (02-forecasts not_fitted.csv): "
        f"{fits or 'none'}; decisions without a valid forecast of their cell's control, "
        f"left out of the cell (scored cells of the test leakage): {cells or 'none'}"
    )


def _cache_models(study_dir: Path, config: StudyConfig) -> pd.DataFrame:
    # models.csv of the forecast cache: release dates and decision spacing (every_hours)
    runs = study_runs(study_dir.parent / "02-forecasts")
    return pd.read_csv(runs / str(config.options["forecast_cache"]) / "models.csv")


def _draw_figure(
    models: pd.DataFrame, config: StudyConfig, verdict: pd.DataFrame, folder: Path
) -> None:
    # fig:rq4: the leakage cells at the figure's horizon, rows in tab:rq1's order (the main
    # family, then the late one, by the forecast cache's release dates; within each, the order
    # of MODEL_NAMES), paper names; writes tables/rq4.csv and figures/rq4.pdf
    options = config.options
    families = figure_families(
        models,
        study_settings(config).scored_folds,
        min_quarters=int(str(options["figure_family_min_quarters"])),
    )
    table = figure_table(verdict, families, horizon=int(str(options["figure_horizon"])))
    table.to_csv(folder / "tables" / "rq4.csv", index=False)
    # the rows of figure_table are already in tab:rq1's order
    order = list(dict.fromkeys(table["model"].astype(str)))
    draw_leakage(
        table.assign(market=table["market"].map(MARKET_LABELS)),
        labels={name: MODEL_NAMES.get(name, name) for name in order},
        market_labels={label: MARKET_NAMES[key] for key, label in MARKET_LABELS.items()},
        x_limit=figure_limit(table),
        pdf_path=folder / "figures" / "rq4.pdf",
        grey_path=folder / "figures" / "rq4_grey.png",
    )


def main() -> None:
    """Apply the decision rule to every cell, answer Q1 and draw fig:rq4."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    settings = study_settings(config)
    not_built = list(leakage_options(config).not_built)
    alpha = settings.alpha
    verdict = leakage_verdict(pd.read_csv(folder / "tables" / "cells.csv"), alpha=alpha)
    verdict.to_csv(folder / "tables" / "verdict.csv", index=False)
    models = _cache_models(invocation.study_dir, config)
    _draw_figure(models, config, verdict, folder)
    hourly = set(models.loc[models["every_hours"] == 0, "model"].astype(str))
    primary = verdict[verdict["test"] == "leakage"]
    scored = primary[primary["status"] == SCORED]
    leaky = primary[primary["shows_leakage"]]
    evidence = (
        f"{len(leaky)} of {len(scored)} scored cells are better before the release than after, "
        "beyond the best baseline: "
        + (_cells(leaky) or "none")
        + "."
        + unscored_note(primary)
        + _unscored_cells(primary)
        + _checks(verdict)
    )
    left_out = _left_out(pd.read_csv(folder / "tables" / "not_fitted.csv"), primary)
    limitations = (
        f"{config.folds} folds; the best baseline, not a model without market data, is the "
        "control, so a change that hits zero-shot models but not baselines also counts; "
        f"one-sided test (H1: skill higher before the release); the interval is "
        f"{settings.bootstrap.level:.0%} two-sided and unadjusted, the decision uses Holm over "
        f"the scored cells of the test leakage; {_size_note(primary, hourly)}; the "
        f"{settings.bootstrap.level:.0%} intervals of fig:rq4 are unadjusted, so an open mark "
        "whose interval excludes 0 is expected now and then; not "
        f"tested (not built on the leakage quarters): {', '.join(not_built) or 'none'}; "
        f"{left_out}"
    )
    files = (
        f"{run}/tables/verdict.csv",
        f"{run}/tables/periods.csv",
        f"{run}/tables/not_fitted.csv",
        f"{run}/tables/rq4.csv",
        f"{run}/figures/rq4.pdf",
    )
    answer = Answer("Q1", "yes" if len(leaky) else "no", evidence, limitations, files)
    path = write_results_readme(invocation.study_dir, [answer], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
