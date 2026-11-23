"""Run 04-rq2-center-vs-scale: the centre and the scale of every cached forecast.

Usage: .venv/bin/python experiments/04-rq2-center-vs-scale/run.py [--dry-run]
[--models NAME ...]
"""

import logging
from pathlib import Path

from lib import centre_scale_rule, centre_scale_tables
from src import LAB_DIR
from src.benchmark import study_cache, study_settings
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("04-rq2-center-vs-scale")


def main() -> None:
    """Score the centre and the scale of the cached forecasts; save the cell and quarter tables."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    cache = study_cache(invocation.study_dir, config)
    rule = centre_scale_rule(config)
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
        tables = centre_scale_tables(
            cache, list(invocation.models), study_settings(config), rule=rule
        )
        run.write_tables(tables)
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
