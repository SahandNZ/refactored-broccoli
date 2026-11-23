import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.experiment._config import StudyConfig, load_study


@dataclass(frozen=True, slots=True)
class Invocation:
    """What a study script was asked to do.

    Attributes
    ----------
    study_dir : Path
        ``experiments/NN-name``.
    config : StudyConfig
        Its parsed ``study.toml``.
    run_id : str
        The run to create (run.py) or read (analyze.py).
    dry_run : bool
        Only check the configuration and print the plan; compute nothing.
    models : tuple of str
        The models to run: those given with ``--models`` (a subset of ``config.models``, for a
        study that builds models one at a time into a run), else every model of the study.
    last_days : int or None
        ``--last-days``: keep only the last that many days of each development fold, for a
        smoke run of a study that builds a cache (02-forecasts); None for the whole folds.
    seeds : tuple of int or None
        ``--seeds``: a subset of ``config.seeds`` for a study that samples them apart
        (02-forecasts' Kronos shards); None for every seed.
    windows : tuple of str or None
        ``--windows``: the fold names whose labels a study that builds a cache forecasts
        (02-forecasts' leakage run); None for every window.
    """

    study_dir: Path
    config: StudyConfig
    run_id: str
    dry_run: bool
    models: tuple[str, ...]
    last_days: int | None
    seeds: tuple[int, ...] | None
    windows: tuple[str, ...] | None


def parse_invocation(description: str | None, *, default_study_dir: Path) -> Invocation:
    """Parse the shared command line of ``run.py`` and ``analyze.py`` and set up logging.

    Options: ``--study-dir`` (default: the script's folder), ``--run-id`` (default: the current
    UTC time, ``YYYY-MM-DDTHHMM``), ``--dry-run``, ``--models`` (default: every model of
    ``study.toml``), ``--last-days`` (development folds only: keep the last N days of each
    fold, for a smoke run), ``--seeds`` (a subset of the seeds, for a sampling model run in
    shards) and ``--windows`` (fold names: the models forecast only these windows' labels).

    Parameters
    ----------
    description : str or None
        Shown by ``--help``: the calling script's ``__doc__``.
    default_study_dir : Path
        The folder of the calling script.

    Returns
    -------
    Invocation
        The parsed arguments with the loaded configuration.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--study-dir", type=Path, default=default_study_dir)
    parser.add_argument("--run-id", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H%M"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--last-days", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--windows", nargs="+")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_study(arguments.study_dir / "study.toml")
    models = tuple(arguments.models or config.models)
    unknown = [model for model in models if model not in config.models]
    if unknown:
        parser.error(f"--models {unknown} are not in study.toml models {list(config.models)}")
    last_days = arguments.last_days
    if last_days is not None and (last_days < 1 or config.folds != "development"):
        parser.error(
            f"--last-days needs development folds and N >= 1, got {last_days}, {config.folds}"
        )
    seeds = None if arguments.seeds is None else tuple(arguments.seeds)
    if seeds is not None and not set(seeds) <= set(config.seeds):
        parser.error(f"--seeds {list(seeds)} are not in study.toml seeds {list(config.seeds)}")
    windows = None if arguments.windows is None else tuple(arguments.windows)
    return Invocation(
        arguments.study_dir,
        config,
        arguments.run_id,
        arguments.dry_run,
        models,
        last_days,
        seeds,
        windows,
    )
