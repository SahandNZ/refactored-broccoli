"""Run 07-rq5-off-hours: weekend crypto and the reopening of closed markets.

Usage: .venv/bin/python experiments/07-rq5-off-hours/run.py [--run-id ID] [--dry-run]
"""

import logging
from pathlib import Path

import pandas as pd

from lib import (
    covariate_use,
    gap_tables,
    off_hours_settings,
    read_weekends,
    reopening_forecasts,
    reopening_tables,
    universe_assets,
)
from src import LAB_DIR
from src.benchmark import benchmark_settings
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("07-rq5-off-hours")


def main() -> None:
    """Find the weekends, forecast the reopenings and test them (Q1), test the gap (Q2)."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    if invocation.models != config.models:
        raise SystemExit(
            "07-rq5-off-hours runs every model of study.toml in one run: drop --models"
        )
    settings = off_hours_settings(config, benchmark_settings(invocation.study_dir, config))
    assets = universe_assets(invocation.study_dir, config, settings)
    if invocation.dry_run:
        logger.info(
            "dry run: %s; %d forecast and %d gap assets of 01-universe run %s, weekends "
            "%s .. %s (forecasts from %s), models %s",
            config.question,
            len(assets.forecast),
            len(assets.gap),
            assets.universe_run,
            settings.start.date(),
            settings.end.date(),
            settings.split_date.date(),
            list(settings.zero_shot),
        )
        return
    data_dir = LAB_DIR / "data"
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir, invocation.run_id, config, data_dir=data_dir, placement=placement
    ) as run:
        weekends = read_weekends(data_dir, assets.gap | assets.forecast, settings)
        table = weekends.table
        tables = gap_tables(table[table["asset"].isin(list(assets.gap))], settings)
        forecasts = reopening_forecasts(weekends, list(assets.forecast), settings)
        forecasts.to_parquet(run.folder / "forecasts.parquet")
        tables |= reopening_tables(forecasts, settings)
        tables["covariate_use"] = covariate_use(forecasts, tables["gap_assets"], settings)
        sets = pd.DataFrame(
            {
                "asset": list(assets.gap),
                "market": [market.value for market in assets.gap.values()],
                "is_forecast": [asset in assets.forecast for asset in assets.gap],
            }
        )
        run.write_tables(tables | {"weekends": table, "asset_sets": sets})
        run.finish(tables["reopening_cells"])


if __name__ == "__main__":
    main()
