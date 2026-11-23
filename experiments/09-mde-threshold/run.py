"""Run 09-mde-threshold: the power of the rq1 rule on the development folds.

Usage: .venv/bin/python experiments/09-mde-threshold/run.py [--run-id ID] [--dry-run]

The rule and its power settings are those of 03-rq1-null (its ``study.toml``, copied into the
run as ``rq1-config.toml``); the folds are always d1-d4.
"""

import dataclasses
import logging
import shutil
from pathlib import Path

from lib import bootstrap_size_table
from src import LAB_DIR
from src.benchmark import null_rule, power_tables, study_cache, study_settings
from src.evaluate import DEVELOPMENT_FOLDS
from src.experiment import Placement, load_study, parse_invocation, start_run

logger = logging.getLogger("09-mde-threshold")


def main() -> None:
    """Plant CRPS gains into every rq1 cell of the development folds and save the MDE tables."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    if config.folds != "development":
        raise SystemExit(f"the threshold is chosen on development folds only, got {config.folds}")
    rq1_toml = invocation.study_dir.parent / "03-rq1-null" / "study.toml"
    rq1 = load_study(rq1_toml)
    settings = dataclasses.replace(study_settings(rq1), scored_folds=DEVELOPMENT_FOLDS)
    cache = study_cache(invocation.study_dir, config)
    if invocation.dry_run:
        logger.info("dry run: %s, cache %s, rule of %s", config.question, cache.folder, rq1_toml)
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
        tables = power_tables(cache, list(rq1.models), settings, null_rule(rq1))
        options = config.options
        size = bootstrap_size_table(
            lengths=tuple(int(length) for length in options["size_lengths"]),
            floors=tuple(int(floor) for floor in options["size_floors"]),
            draws=int(options["size_draws"]),
            resamples=int(options["size_resamples"]),
            seed=int(options["size_seed"]),
        )
        run.write_tables({name: tables[name] for name in ("cells", "power", "mde")})
        run.write_tables({"bootstrap_size": size})
        run.finish(tables["mde"])


if __name__ == "__main__":
    main()
