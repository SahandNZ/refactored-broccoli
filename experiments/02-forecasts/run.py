"""Run 02-forecasts: the shared forecast cache of every model and baseline.

Usage: .venv/bin/python experiments/02-forecasts/run.py [--run-id ID]
[--models NAME ...] [--last-days N] [--seeds S ...] [--windows W ...] [--dry-run]

A new run id starts a run; the id of an existing run resumes it with the run's own
config.toml (refused if the git commit, the dirty flag, the plan or the data changed since it
started): the listed models (default: all of study.toml) are built into it, models already
written are skipped, and the new models' rows are added to metrics.parquet. --last-days N
(development folds only, a smoke run) keeps the last N days of each fold; --seeds S ...
samples only those seeds of Kronos (a shard; its forecasts are written once every seed's
samples are in the run's samples/ folder); --windows W ... (fold names) forecasts only the
labels of those windows (the labels of every window are still written), e.g. the leakage
quarters of a run whose other windows another run holds.
"""

import logging
from pathlib import Path

from src import LAB_DIR
from src.benchmark import CachePart, build_cache
from src.experiment import (
    Placement,
    StudyConfig,
    parse_invocation,
    resume_run,
    start_run,
    study_runs,
)

logger = logging.getLogger("02-forecasts")


def _universe(study_dir: Path, config: StudyConfig) -> Path:
    # the universe table of the 01-universe run named in the options
    run = study_runs(study_dir.parent / "01-universe") / str(config.options["universe_run"])
    return run / "tables" / "universe.csv"


def main() -> None:
    """Build the labels and the forecasts of the listed models into the run folder."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    study_dir, run_id = invocation.study_dir, invocation.run_id
    data_dir = LAB_DIR / "data"
    if invocation.dry_run:
        universe = _universe(study_dir, invocation.config)
        logger.info("dry run: models %s, universe %s", invocation.models, universe)
        return
    is_resumed = (study_runs(study_dir) / run_id).exists()
    if is_resumed:
        # meta.json keeps the record of the first start: only the same code, plan and data
        # resume, and the new models' rows are added to metrics.parquet
        run = resume_run(study_dir, run_id, data_dir=data_dir)
    else:
        placement = Placement("local", None, None, 1)
        run = start_run(
            study_dir, run_id, invocation.config, data_dir=data_dir, placement=placement
        )
    with run:
        built = build_cache(
            run.folder,
            run.config,
            data_dir=data_dir,
            universe=_universe(study_dir, run.config),
            models=invocation.models,
            part=CachePart(
                last_days=invocation.last_days,
                seeds=invocation.seeds,
                windows=invocation.windows,
            ),
        )
        if is_resumed:
            run.append_metrics(built)
        else:
            run.finish(built)


if __name__ == "__main__":
    main()
