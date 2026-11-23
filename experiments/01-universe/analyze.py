"""Analyse a run of 01-universe: the correlation figure, Q3 and the results README.

Q3 ranks the models of study.toml on the test forecast cache of 02-forecasts named in
``[options.ranking]`` (no forecast is made) and writes ``tables/ranking_*.csv`` into the run.

Usage: .venv/bin/python experiments/01-universe/analyze.py --run-id ID
"""

import json
import logging
from pathlib import Path

import pandas as pd

from lib import (
    BOOTSTRAP_RANDOM_UNIVERSES,
    RankingSettings,
    ranking_tables,
    read_ranking_settings,
    read_settings,
    split_summary,
    stability_answer,
    universe_beats_alternatives,
)
from src.benchmark import read_cache
from src.experiment import Answer, Invocation, parse_invocation, study_runs, write_results_readme
from src.figures import draw_correlation

logger = logging.getLogger("01-universe")


def main() -> None:
    """Draw the universe's correlation heatmap and answer the questions of the plan."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    settings = read_settings(invocation.config.options)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    universe = pd.read_csv(folder / "tables" / "universe.csv")
    diversity = pd.read_csv(folder / "tables" / "diversity.csv")
    correlation = pd.read_parquet(folder / "correlation.parquet")
    chosen = universe[universe["in_universe"]].sort_values(["market", "pool", "order"])
    symbols = chosen["symbol"].tolist()
    matrix = correlation.loc[symbols, symbols]
    matrix.to_csv(folder / "tables" / "universe_correlation.csv")
    draw_correlation(
        matrix,
        chosen["market"].tolist(),
        pdf_path=folder / "figures" / "universe_correlation.pdf",
        grey_path=folder / "figures" / "universe_correlation_grey.png",
    )
    selection = diversity[diversity["window"] == "selection"]
    holdout = diversity[diversity["window"] == "holdout"]
    # Q1 is decided on the held-out window; the selection window, where the universe was
    # picked, is reported as the in-sample number
    answer, evidence = universe_beats_alternatives(holdout, percentile=settings.q1_percentile)
    in_sample_answer, in_sample_evidence = universe_beats_alternatives(
        selection, percentile=settings.q1_percentile
    )
    evidence = (
        f"Held-out window {settings.end} to {settings.holdout_end} (the decision): {evidence} "
        f"In-sample, on the selection window {settings.start} to {settings.end} where the "
        f"universe was picked (not the decision): {in_sample_evidence} The rule would answer "
        f"{in_sample_answer} there."
    )
    stocks = universe[(universe["market"] == "stock") & universe["pool"].notna()]
    non_us = int((stocks["pool"] == "stock_non_us").sum())
    limitations = (
        "the candidate lists were drawn up in 2026 (crypto perpetuals still trading then, "
        "today's large-cap stocks), a survivorship bias of the candidate set that mostly "
        "affects return-based results; coverage to the end of the selection window drops "
        "assets that stopped trading inside it (it uses only data available at selection "
        "time); the held-out window is short (10 months, about 43 weekly returns, so its matrix "
        "has rank at most 43 and effective-rank levels are compared only within a window) "
        "and the bootstrap interval of the gap is that of the selection window, in-sample, "
        "and the normal interval (gap +- z times the standard deviation of the paired "
        f"bootstrap gaps; its Random median uses the first {BOOTSTRAP_RANDOM_UNIVERSES} draws), "
        "because the percentile interval is shifted below the point gap by the in-sample "
        "optimism and the repeated weeks (resampling lowers the tuned universe more); trading "
        f"hours differ by market; {len(stocks)} stock candidates pass the filters, {non_us} of "
        "them non-US"
    )
    files = (
        f"{run}/tables/universe.csv",
        f"{run}/tables/diversity.csv",
        f"{run}/tables/universe_correlation.csv",
        f"{run}/figures/universe_correlation.pdf",
        f"{run}/tables/sector_mix.csv",
        f"{run}/tables/comparison_universes.csv",
    )
    answers = [
        Answer("Q1", answer, evidence, limitations, files),
        _ranking_answer(invocation, universe),
    ]
    path = write_results_readme(invocation.study_dir, answers, run_ids=[run])
    logger.info("wrote %s", path)


def _data_version(folder: Path) -> str:
    # the data version a run recorded in its meta.json (hash of data/manifest.json)
    return str(json.loads((folder / "meta.json").read_text())["data_version"])


def _group_sentence(tables: dict[str, pd.DataFrame], group: str, ranking: RankingSettings) -> str:
    # the evidence of one group: its split taus and agreement, its market and quarter taus
    splits = tables["ranking_splits"]
    splits = splits[splits["group"] == group]
    parts = tables["ranking_parts"]
    parts = parts[parts["group"] == group]
    scores = tables["ranking_scores"]
    scores = scores[scores["group"] == group]
    median, low, high = split_summary(splits, group, level=ranking.range_level)
    markets = parts.loc[parts["check"] == "market", "tau"]
    quarters = parts.loc[parts["check"] == "quarter", "tau"]
    top = str(scores.loc[scores["rank"] == 1, "model"].iloc[0])
    return (
        f"{len(scores)} models, {int(scores['decisions'].iloc[0]):,} decisions: Kendall tau of "
        f"the two halves of {len(splits)} random splits median {median:.2f} (central "
        f"{100 * ranking.range_level:.0f} % of splits {low:.2f} to {high:.2f}); the same first "
        f"model in both halves in {100 * splits['same_top'].mean():.1f} % of splits and the "
        f"same first three in order in {100 * splits['same_top3'].mean():.1f} % (first on the "
        f"whole universe: {top}); each market against the other four {markets.min():.2f} to "
        f"{markets.max():.2f}; each test quarter against the other three {quarters.min():.2f} "
        f"to {quarters.max():.2f}."
    )


def _ranking_answer(invocation: Invocation, universe: pd.DataFrame) -> Answer:
    # Q3: the ranking tables of the run, written into it, and the answer they give
    ranking: RankingSettings = read_ranking_settings(invocation.config.options)
    cache_folder = study_runs(invocation.study_dir.parent / "02-forecasts") / ranking.forecast_cache
    run = invocation.run_id
    run_folder = study_runs(invocation.study_dir) / run
    if _data_version(cache_folder) != _data_version(run_folder):
        raise ValueError(
            f"the data versions of {cache_folder} and {run_folder} differ (experiments rule 10)"
        )
    cache = read_cache(cache_folder)
    listed = universe[universe["in_universe"]]
    pools = pd.Series(listed["pool"].to_numpy(), index=listed["symbol"].to_numpy())
    tables = ranking_tables(
        cache, list(invocation.config.models), pools, ranking, seed=invocation.config.seeds[0]
    )
    for name, table in tables.items():
        table.to_csv(run_folder / "tables" / f"{name}.csv", index=False)
    answer, _ = stability_answer(tables["ranking_splits"], ranking)
    labels = {"hourly": "Hourly models", "grid": "All models on Kronos's decision points"}
    groups = [group for group in labels if group in set(tables["ranking_scores"]["group"])]
    evidence = (
        f"Computed by analyze.py from the test forecast cache {ranking.forecast_cache} of "
        f"02-forecasts (same data version as this run), not by the selection run whose commit "
        f"the table above shows: test quarters {', '.join(ranking.windows)}, horizon "
        f"{ranking.horizon}, mean CRPS per asset averaged over the assets, on the decisions "
        f"every compared model scored; decided by the hourly group's median split tau against "
        f"{ranking.tau_threshold}. "
        + " ".join(
            f"{labels[group]} ({_group_sentence(tables, group, ranking)})" for group in groups
        )
    )
    limitations = (
        "exploratory, amended after the plan lock and after RQ1's test results were seen "
        "(Plan changes, 2026-11-18); parts of the one universe, not other universes (the "
        "Liquid and Random comparison needs new forecasts and is left to later work); with 11 "
        "models a tau of 0.7 allows 8 of the 55 pairs to swap (the plan's four models needed "
        "the same order), and most pairs are far apart, so the first-model and first-three "
        "shares are the sharper check; the halves share the test hours and the market-wide "
        "moves, so their rankings are not independent; the range is the spread over random "
        "splits, not a confidence interval"
    )
    files = tuple(f"{run}/tables/{name}.csv" for name in tables)
    return Answer("Q3", answer, evidence, limitations, files)


if __name__ == "__main__":
    main()
