"""Analyse a run of 05-rq3-net: the verdict and the results README.

The bodies of the paper's tables come from ``make_tab_rq3_break_even.py`` on the same run.

Usage: .venv/bin/python experiments/05-rq3-net/analyze.py --run-id ID
"""

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from lib import BUY_AND_HOLD, LATE, MAIN, net_verdict
from src.benchmark import SCORED, study_settings, unscored_note
from src.experiment import Answer, load_study, parse_invocation, study_runs, write_results_readme

logger = logging.getLogger("05-rq3-net")


def _bps(value: float) -> str:
    # one decimal in bps, or "--" for an undefined value
    return "--" if math.isnan(value) else f"{value:.1f}"


def _p_value(value: float) -> str:
    # two significant digits; below 0.001 written as <0.001 (tables.md 5)
    return "<0.001" if value < 0.001 else f"{value:.2g}"


def _earning(verdict: pd.DataFrame) -> str:
    # the cells of `verdict` that earn money after costs, with their numbers
    names = [
        f"{row['model']} {row['market']} {row['horizon']}h (net Sharpe {row['net_sharpe']:.2f}, "
        f"mean net return {_bps(row['net_mean'] * 1e4)} bps per book time "
        f"[{_bps(row['net_low'] * 1e4)}, {_bps(row['net_high'] * 1e4)}], "
        f"adjusted p {_p_value(row['adjusted_p'])}, break-even {_bps(row['break_even_bps'])} bps "
        f"[{_bps(row['break_even_low'])}, {_bps(row['break_even_high'])}] vs cost "
        f"{_bps(row['cost_bps'])} bps)"
        for row in verdict[verdict["earns_after_costs"]].to_dict("records")
    ]
    scored = int((verdict["status"] == SCORED).sum())
    return f"{len(names)} of {scored} scored cells earn money after costs: " + (
        "; ".join(names) or "none"
    )


def _reference(cells: pd.DataFrame, *, model: str, horizon: int) -> str:
    # a reference book's (ridge regression's, buy and hold's) break-even and net mean
    # against the cost paid, per market
    rows = cells[(cells["model"] == model) & (cells["horizon"] == horizon)]
    parts = [
        f"{row['market']} {_bps(row['break_even_bps'])} bps vs cost {_bps(row['cost_bps'])} bps "
        f"(net {_bps(row['net_mean'] * 1e4)} bps per book time)"
        for row in rows.to_dict("records")
    ]
    return f"{model} (reference), break-even at {horizon}h: " + "; ".join(parts)


def _best_break_even(verdict: pd.DataFrame, *, horizon: int) -> str:
    # per market, the scored zero-shot cell with the highest break-even at `horizon` (either
    # family, named), with its interval, against the cost it pays
    scored = verdict[(verdict["status"] == SCORED) & (verdict["horizon"] == horizon)]
    scored = scored.dropna(subset=["break_even_bps"])
    best = scored.sort_values("break_even_bps").groupby("market").tail(1)
    parts = [
        f"{row['market']} {row['model']} ({row['family']} family) "
        f"{_bps(row['break_even_bps'])} bps "
        f"[{_bps(row['break_even_low'])}, {_bps(row['break_even_high'])}] vs cost "
        f"{_bps(row['cost_bps'])} bps"
        for row in best.sort_values("market").to_dict("records")
    ]
    return f"Highest break-even per market at {horizon}h: " + ("; ".join(parts) or "none")


def _delay(verdict: pd.DataFrame, delayed: pd.DataFrame) -> str:
    # the one-bar delay check against the scored zero-shot cells traded on time
    keys = ["model", "market", "horizon"]
    scored = verdict.loc[verdict["status"] == SCORED, [*keys, "net_mean", "cost_bps"]]
    late = delayed[[*keys, "net_mean", "break_even_bps"]].rename(
        columns={"net_mean": "net_late", "break_even_bps": "break_even_late"}
    )
    paired = scored.merge(late, on=keys)
    lower = int((paired["net_late"] < paired["net_mean"]).sum())
    above = int((paired["break_even_late"] > paired["cost_bps"]).sum())
    return (
        f"One-bar delay (robustness, delayed.csv): in {len(paired)} scored cells with a delayed "
        f"book, the mean net return is lower with the delay in {lower}, and the delayed "
        f"break-even is above the on-time cell's cost in {above}"
    )


def _texts(
    verdict: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    options: dict[str, Any],
    *,
    alpha: float,
    folds: str,
) -> tuple[str, str]:
    # the evidence and the limitations of Q1's answer, from the verdict and the run's
    # "cells" and "delayed" tables
    cells = tables["cells"]
    horizon = int(options["table_horizon"])
    main_family = verdict[verdict["family"] == MAIN]
    late_family = verdict[verdict["family"] == LATE]
    scored = verdict[verdict["status"] == SCORED]
    above = int((scored["break_even_bps"] > scored["cost_bps"]).sum())
    flagged = int(verdict["has_few_blocks"].sum())
    low = int(verdict["is_low_turnover"].sum())
    references = ". ".join(
        _reference(cells, model=model, horizon=horizon)
        for model in (str(options["reference"]), BUY_AND_HOLD)
    )
    evidence = (
        f"Main family: {_earning(main_family)}. Late-release family (own verdict): "
        f"{_earning(late_family)}. The break-even cost per side is above the cost paid in "
        f"{above} of {len(scored)} scored cells (every zero-shot cell's break-even and its "
        "interval are in verdict.csv, the baselines' cells in cells.csv; per-quarter means in "
        "quarters.csv; the one-bar delay check in delayed.csv). "
        f"{_best_break_even(verdict, horizon=horizon)}. {references}. "
        f"{_delay(verdict, tables['delayed'])}. {flagged} cells have fewer than "
        f"{options['bootstrap_min_blocks']} bootstrap blocks (has_few_blocks); {low} cells "
        f"trade less than {options['min_turnover']} per effective time (is_low_turnover, "
        "close to buy and hold)." + unscored_note(verdict)
    )
    limitations = (
        f"{folds} folds; only decisions after each model's release; equal weight 1/n "
        "over each market's assets (an asset without a position is cash; the stock market "
        "pools US and non-US stocks), each position held for the horizon in the asset's own "
        "bars; fixed costs per side, the final closing trade charged, no impact, funding or "
        f"delay (delay only in the robustness check); one-sided tests at {alpha}, Holm per "
        f"family over all horizons and markets; the intervals are {1 - alpha:.0%} unadjusted; "
        "US stock costs are an assumed 1 bp half spread"
    )
    return evidence, limitations


def main() -> None:
    """Apply the decision rule to every zero-shot cell and answer Q1."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    run = invocation.run_id
    folder = study_runs(invocation.study_dir) / run
    config = load_study(folder / "config.toml")
    options: dict[str, Any] = config.options
    alpha = study_settings(config).alpha
    if int(options["table_horizon"]) != 1:
        raise ValueError(
            "table_horizon must be 1 (buy and hold is at horizon 1), got "
            f"{options['table_horizon']}"
        )
    cells = pd.read_csv(folder / "tables" / "cells.csv")
    verdict = net_verdict(
        cells[cells["kind"] == "zero-shot"],
        alpha=alpha,
        main_min_quarters=int(options["main_min_quarters"]),
        min_blocks=float(options["bootstrap_min_blocks"]),
        min_turnover=float(options["min_turnover"]),
    )
    tables = folder / "tables"
    verdict.to_csv(tables / "verdict.csv", index=False)
    references = cells["model"].isin([str(options["reference"]), BUY_AND_HOLD])
    cells[references].to_csv(tables / "reference.csv", index=False)
    tables = {"cells": cells, "delayed": pd.read_csv(folder / "tables" / "delayed.csv")}
    evidence, limitations = _texts(verdict, tables, options, alpha=alpha, folds=config.folds)
    files = tuple(
        f"{run}/tables/{name}"
        for name in (
            "verdict.csv",
            "cells.csv",
            "quarters.csv",
            "reference.csv",
            "delayed.csv",
            "costs.csv",
        )
    )
    earns = bool(verdict.loc[verdict["family"] == MAIN, "earns_after_costs"].any())
    answer = Answer("Q1", "yes" if earns else "no", evidence, limitations, files)
    path = write_results_readme(invocation.study_dir, [answer], run_ids=[run])
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
