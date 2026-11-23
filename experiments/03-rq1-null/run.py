"""Run 03-rq1-null: the CRPS of every zero-shot model against the best baseline, and its power.

Usage: .venv/bin/python experiments/03-rq1-null/run.py [--run-id ID] [--dry-run]
"""

import logging
from pathlib import Path

from src import LAB_DIR
from src.benchmark import null_rule, power_tables, study_cache, study_settings
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("03-rq1-null")


def main() -> None:
    """Score the cached forecasts and save the cell, quarter, power and MDE tables."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    cache = study_cache(invocation.study_dir, config)
    if invocation.dry_run:
        logger.info("dry run: %s, cache %s, folds %s", config.question, cache.folder, config.folds)
        return
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir,
        invocation.run_id,
        config,
        data_dir=LAB_DIR / "data",
        placement=placement,
    ) as run:
        tables = power_tables(cache, list(config.models), study_settings(config), null_rule(config))
        run.write_tables({name: tables[name] for name in ("cells", "quarters", "power", "mde")})
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
