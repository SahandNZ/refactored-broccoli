"""Run 06-rq4-leakage: each model's skill before and after its release, against the best baseline.

Usage: .venv/bin/python experiments/06-rq4-leakage/run.py [--run-id ID] [--dry-run]

With the locked plan (test folds) "before" is the leakage quarters l1-l12 and "after" t1-t4;
while the plan is open both come from d1-d4 (a check of the code, not a leakage estimate). The
leakage quarters may come from a second 02-forecasts run (options.leakage_cache).
"""

import logging
from pathlib import Path

from lib import JoinedCache, leakage_cache, leakage_options, leakage_tables, not_fitted
from src import LAB_DIR
from src.benchmark import study_settings
from src.evaluate import LEAKAGE_FOLDS
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("06-rq4-leakage")


def main() -> None:
    """Score the cached forecasts and save the period and cell tables (every test of every cell)."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    cache = leakage_cache(invocation.study_dir, config)
    settings = study_settings(config)
    before = LEAKAGE_FOLDS if config.folds == "test" else settings.scored_folds
    if invocation.dry_run:
        leakage_run = cache.other.folder if isinstance(cache, JoinedCache) else None
        logger.info(
            "dry run: %s, cache %s, leakage run %s, folds %s",
            config.question,
            cache.folder,
            leakage_run,
            config.folds,
        )
        return
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir,
        invocation.run_id,
        config,
        data_dir=LAB_DIR / "data",
        placement=placement,
    ) as run:
        tables = leakage_tables(
            cache,
            list(config.models),
            settings,
            before_folds=before,
            options=leakage_options(config),
        )
        tables["not_fitted"] = not_fitted(invocation.study_dir, config)
        run.write_tables({name: tables[name] for name in ("cells", "periods", "not_fitted")})
        run.finish(tables["cells"])


if __name__ == "__main__":
    main()
