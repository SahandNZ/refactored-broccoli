"""Run 08-closures: every model after a reopening, from every representation of the closed hours.

Usage: .venv/bin/python experiments/08-closures/run.py [--run-id ID] [--dry-run]
[--models NAME ...]

``--models`` runs a subset of the study's models, only on the development folds: the test run
(``[folds] use = "test"``) forecasts every model of study.toml in one run.
"""

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from lib import BASELINE, benchmark_assets, closure_losses, closure_settings
from src import LAB_DIR
from src.benchmark import benchmark_settings
from src.enums import Timeframe
from src.experiment import Placement, parse_invocation, start_run
from src.panel import hourly_closes, load_panel_bars

logger = logging.getLogger("08-closures")


def main() -> None:
    """Forecast and score every decision after a reopening; save the losses and the summary."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config, study_dir = invocation.config, invocation.study_dir
    if len(set(invocation.models)) != len(invocation.models):
        raise SystemExit(f"--models names a model twice: {list(invocation.models)}")
    if invocation.last_days is not None:
        raise SystemExit("08-closures scores whole folds: drop --last-days")
    if set(invocation.models) != set(config.models) and config.folds == "test":
        raise SystemExit("the test run of 08-closures forecasts every model: drop --models")
    # a subset keeps the baseline, which every results table holds (experiments rule 5)
    models = invocation.models
    if BASELINE in config.models and BASELINE not in models:
        models = (*models, BASELINE)
    settings = closure_settings(
        replace(config, models=models), benchmark_settings(study_dir, config)
    )
    markets = benchmark_assets(study_dir, config)
    if invocation.dry_run:
        logger.info(
            "dry run: %s; %d assets, models %s", config.question, len(markets), settings.models
        )
        return
    data_dir = LAB_DIR / "data"
    placement = Placement("local", None, None, 1)
    with start_run(
        study_dir, invocation.run_id, config, data_dir=data_dir, placement=placement
    ) as run:
        logger.info("models: %s", settings.models)
        closes = hourly_closes(load_panel_bars(data_dir, markets, (Timeframe.H1,)))
        tables = closure_losses(closes, markets, settings)
        # the models this run forecast (config.toml is the whole study.toml): a rerun of a
        # subset passes them as --models
        models = pd.DataFrame({"model": list(settings.models)})
        run.write_tables({"summary": tables["summary"], "models": models})
        run.finish(tables["losses"])


if __name__ == "__main__":
    main()
