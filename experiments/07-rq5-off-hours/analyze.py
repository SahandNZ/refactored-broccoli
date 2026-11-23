"""Analyse a run of 07-rq5-off-hours and write the results README.

Usage: .venv/bin/python experiments/07-rq5-off-hours/analyze.py --run-id ID
"""

import logging
from pathlib import Path

import pandas as pd

from lib import OffHoursSettings, off_hours_settings
from src.benchmark import benchmark_settings
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("07-rq5-off-hours")


def _verdict(found: pd.DataFrame, judged: pd.DataFrame) -> str:
    # yes when a judged row says yes, no when rows were judged, inconclusive without any
    if len(found):
        return "yes"
    if len(judged):
        return "no"
    return "inconclusive"


def _forecast_answer(tables: Path, settings: OffHoursSettings, run: str) -> Answer:
    cells = pd.read_csv(tables / "reopening_cells.csv")
    tests = pd.read_csv(tables / "reopening_tests.csv")
    use = pd.read_csv(tables / "covariate_use.csv")
    tested = cells[cells["status"] == "scored"]
    best = tested[tested["covariate_is_best"]]
    per_test = [
        f"{arm} vs {reference}: {int(group['is_better'].sum())} of {len(group)} cells, median "
        f"skill {group['skill'].median():+.1f} % (cell intervals from "
        f"{group['skill_low'].min():+.1f} to {group['skill_high'].max():+.1f} %)"
        for (arm, reference), group in tests.groupby(["arm", "reference"], sort=False)
    ]
    shares = use.groupby("model")["use_share"].median()
    evidence = (
        f"the covariate arm beats all four other arms in {len(best)} of {len(tested)} tested "
        "market and model cells: "
        + (
            "; ".join(f"{m} {o}" for m, o in zip(best["market"], best["model"], strict=True))
            or "none"
        )
        + ". One-sided tests (adjusted p below "
        f"{settings.alpha}): "
        + "; ".join(per_test)
        + ". Median share of the move's crypto slope a model passes on from its covariate "
        "(descriptive): "
        + "; ".join(f"{model} {share:.2f}" for model, share in shares.items())
        + f". {len(cells) - len(tested)} cells not tested."
    )
    limitations = (
        f"weekends from {settings.split_date.date()}, scored in the "
        f"{settings.scored_folds[0].name}-{settings.scored_folds[-1].name} folds, about one "
        "weekly point per weekend and cell (low power); no cut at a model's release: some or "
        "all scored weekends precede a model's release (on d1-d4 all of them, for every "
        "model; on t1-t4 none for Chronos-2, until 2026-07-02 for TiRex-2 and almost all "
        "for t0, released 2026-10-15), so a model that memorised a reopening could favour "
        "its own arms over the correction and the model-free arm; one forecast per weekend "
        "and model; the label is the close of the first bar, not the gap of Q2; the "
        "correction and the model-free arm "
        "are refitted before every weekend on all earlier weekends since "
        f"{settings.split_date.date()} whose label was known (at least "
        f"{settings.min_fit_weekends}), also inside the scored quarter (past labels only); "
        "the covariate use is descriptive (no interval)"
    )
    files = tuple(
        f"{run}/tables/{name}.csv"
        for name in ("reopening_cells", "reopening_tests", "covariate_use")
    )
    return Answer("Q1", _verdict(best, tested), evidence, limitations, files)


def _gap_answer(tables: Path, settings: OffHoursSettings, run: str) -> Answer:
    assets = pd.read_csv(tables / "gap_assets.csv")
    tested = assets[assets["status"] == "tested"]
    yes = tested[tested["is_yes"]]
    names = [
        f"{asset} beta {beta:.2f} [{low:.2f}, {high:.2f}] ({sign}, adjusted p {p:.2g}), "
        f"trimmed rank corr {trimmed:.2f}, R^2 +{gain:.2f}"
        for asset, beta, low, high, sign, p, trimmed, gain in zip(
            yes["asset"],
            yes["beta"],
            yes["beta_low"],
            yes["beta_high"],
            yes["sign"],
            yes["adjusted_p"],
            yes["rank_corr_trimmed"],
            yes["r2_increment"],
            strict=True,
        )
    ]
    sides = pd.read_csv(tables / "gap_directions.csv")
    sides = sides[sides["status"] == "tested"].groupby("direction")["rank_corr"].median()
    halves = pd.read_csv(tables / "gap_half_years.csv")
    halves = halves[halves["status"] == "tested"]
    evidence = (
        f"{len(yes)} of {len(tested)} tested assets: "
        + ("; ".join(names) or "none")
        + f". Median rank correlation {tested['rank_corr'].median():.2f}; on crypto-down "
        f"weekends {sides.get('down', float('nan')):.2f}, on crypto-up weekends "
        f"{sides.get('up', float('nan')):.2f}; positive in {int((halves['rank_corr'] > 0).sum())}"
        f" of {len(halves)} tested asset half-years. {len(assets) - len(tested)} assets "
        "not tested."
    )
    limitations = (
        f"weekends {settings.start.date()} .. {settings.end.date()}, in effect from the "
        "listing of the last basket asset on Binance USD-M, before which the basket move is "
        "missing (weekends counted as incomplete); "
        "assets that passed 01-universe's data checks of 2020-2023, but from candidate lists "
        f"of 2026; Friday's information is only the last {settings.friday_hours} hours of the "
        f"asset and the basket; two-sided HAC t, Holm over {len(tested)} assets, a little "
        f"liberal with {settings.hac_lags} Bartlett lags (about 7 % at 5 % with 4 lags on 100 "
        "synthetic null weekends); "
        f"{settings.interval_level:.0%} HAC intervals, not Holm-adjusted; the gap is a mid "
        "quote of a CFD, whose spread is wide at the reopening; an in-sample regression over "
        "all weekends up to the end of the scored folds, so a test run includes the "
        "development months"
    )
    files = tuple(
        f"{run}/tables/{name}.csv" for name in ("gap_assets", "gap_half_years", "gap_directions")
    )
    return Answer("Q2", _verdict(yes, tested), evidence, limitations, files)


def main() -> None:
    """Apply the decision rules of Q1 and Q2 and write the results README."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    settings = off_hours_settings(config, benchmark_settings(invocation.study_dir, config))
    answers = [
        _forecast_answer(folder / "tables", settings, run),
        _gap_answer(folder / "tables", settings, run),
    ]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
