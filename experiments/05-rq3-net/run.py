"""Run 05-rq3-net: gross and net result of every model's sign book and its break-even cost.

Usage: .venv/bin/python experiments/05-rq3-net/run.py [--run-id ID] [--dry-run]
"""

import logging
from pathlib import Path

from lib import net_tables
from src import LAB_DIR
from src.benchmark import study_cache, study_settings
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("05-rq3-net")


def main() -> None:
    """Trade the cached medians and save the cell, quarter, delay-check and cost tables."""
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
        tables = net_tables(cache, list(config.models), study_settings(config))
        run.write_tables(tables)
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
