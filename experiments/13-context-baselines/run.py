"""Run 13-context-baselines: rq1's test at 1 bar against baselines fitted on the context (Q1).

Usage: .venv/bin/python experiments/13-context-baselines/run.py [--run-id ID]
[--dry-run]

The rule (alpha, quarters, bootstrap, skill draws, the best baseline chosen on d1-d4) is that
of 03-rq1-null (its ``study.toml``, copied into the run as ``rq1-config.toml``); the random
walk, climatology, GARCH(1,1) and the ridge regression are refitted on the last ``window``
own bars before each decision; EWMA-normal fits nothing and is read from the cache.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from lib import ContextSettings, context_cache, context_forecasts
from src import LAB_DIR
from src.benchmark import null_rule, null_tables, study_cache, study_settings
from src.evaluate import DEVELOPMENT_FOLDS, TEST_FOLDS
from src.experiment import Placement, load_study, parse_invocation, start_run

logger = logging.getLogger("13-context-baselines")


def main() -> None:
    """Refit the baselines on the context, score every model at 1 bar and save the tables."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    options: dict[str, Any] = config.options
    rq1_toml = invocation.study_dir.parent / "03-rq1-null" / "study.toml"
    rq1 = load_study(rq1_toml)
    settings = ContextSettings(
        window=int(options["window"]),
        lags=tuple(int(lag) for lag in options["lags"]),
        ridge=float(options["ridge"]),
        workers=int(options["workers"]),
    )
    cache = study_cache(invocation.study_dir, config)
    windows = [fold.name for fold in (*DEVELOPMENT_FOLDS, *TEST_FOLDS)]
    if invocation.dry_run:
        logger.info("dry run: %s, cache %s, %s", config.question, cache.folder, settings)
        return
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir,
        invocation.run_id,
        config,
        data_dir=LAB_DIR / "data",
        placement=placement,
    ) as run:
        shutil.copy(rq1_toml, run.folder / "rq1-config.toml")
        context, failed = context_forecasts(cache, windows=windows, settings=settings)
        view = context_cache(cache, context, windows)
        tables = null_tables(view, list(config.models), study_settings(rq1), null_rule(rq1))
        run.write_tables({"cells": tables["cells"], "quarters": tables["quarters"]})
        run.write_tables({"garch_not_fitted": failed})
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
