"""Run 12-horizon-spans: the calendar span of each horizon per market (Q1).

Usage: .venv/bin/python experiments/12-horizon-spans/run.py [--run-id ID] [--dry-run]
"""

import logging
from pathlib import Path
from typing import Any

from lib import horizon_spans
from src import LAB_DIR
from src.benchmark import study_cache
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("12-horizon-spans")


def main() -> None:
    """Measure the spans on the labels of the forecast cache and save the table."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    cache = study_cache(invocation.study_dir, config)
    options: dict[str, Any] = config.options
    windows = [str(window) for window in options["windows"]]
    if invocation.dry_run:
        logger.info("dry run: %s, cache %s, windows %s", config.question, cache.folder, windows)
        return
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir,
        invocation.run_id,
        config,
        data_dir=LAB_DIR / "data",
        placement=placement,
    ) as run:
        spans = horizon_spans(cache, windows=windows)
        run.write_tables({"spans": spans})
        run.finish(spans)


if __name__ == "__main__":
    main()
