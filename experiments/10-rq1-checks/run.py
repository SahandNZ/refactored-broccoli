"""Run 10-rq1-checks: rq1's test with every zero-shot model on Kronos's decision points (Q1).

Also the gain of each hourly model over each Kronos checkpoint on the same decision points,
and the share of decisions at a reopening.

Usage: .venv/bin/python experiments/10-rq1-checks/run.py [--run-id ID] [--dry-run]

The rule (alpha, quarters, bootstrap, skill draws) is that of 03-rq1-null (its ``study.toml``,
copied into the run as ``rq1-config.toml``); the scored folds are its test folds.
"""

import logging
import shutil
from pathlib import Path

from lib import grid_cache, kronos_gaps, reopening_shares
from src import LAB_DIR
from src.benchmark import null_rule, null_tables, study_cache, study_settings
from src.experiment import Placement, load_study, parse_invocation, start_run

logger = logging.getLogger("10-rq1-checks")


def main() -> None:
    """Score every model on the grid model's decisions and count the reopening decisions."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    rq1_toml = invocation.study_dir.parent / "03-rq1-null" / "study.toml"
    rq1 = load_study(rq1_toml)
    settings = study_settings(rq1)
    grid = str(config.options["grid_model"])
    cache = study_cache(invocation.study_dir, config)
    if invocation.dry_run:
        logger.info("dry run: %s, cache %s, grid %s", config.question, cache.folder, grid)
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
        view = grid_cache(cache, grid)
        tables = null_tables(view, list(config.models), settings, null_rule(rq1))
        windows = [fold.name for fold in settings.scored_folds]
        every = dict(zip(cache.models["model"], cache.models["every_hours"], strict=True))
        kinds = dict(zip(cache.models["model"], cache.models["kind"], strict=True))
        zero_shot = [model for model in config.models if kinds[model] == "zero-shot"]
        gaps = kronos_gaps(
            view,
            [model for model in zero_shot if int(every[model]) == 0],
            [model for model in zero_shot if int(every[model]) > 0],
            windows=windows,
            horizon=int(str(config.options["gap_horizon"])),
            bootstrap=settings.bootstrap,
        )
        closure = float(str(config.options["min_closure_hours"]))
        reopening = reopening_shares(cache, grid, windows=windows, min_closure_hours=closure)
        run.write_tables({"cells": tables["cells"], "quarters": tables["quarters"]})
        run.write_tables({"reopening": reopening, "kronos_gaps": gaps})
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
