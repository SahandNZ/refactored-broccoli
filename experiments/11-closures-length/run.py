"""Run 11-closures-length: the benchmark input against trading hours only after a closure (Q1).

Usage: .venv/bin/python experiments/11-closures-length/run.py [--run-id ID] [--dry-run]

Reads the per-decision scores of the 08-closures run named by ``closures_run`` (its
``metrics.parquet``), its settings (``config.toml``, copied into the run as
``closures-config.toml``) and its Holm families (its committed ``verdict.csv``); no forecast
is made.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

from lib import length_by_market, length_tests, read_length_settings
from src import LAB_DIR
from src.experiment import Placement, load_study, parse_invocation, start_run, study_runs

logger = logging.getLogger("11-closures-length")


def main() -> None:
    """Test the context's length at the same encoding and save its tables."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    closures = invocation.study_dir.parent / "08-closures"
    source = str(config.options["closures_run"])
    folder = study_runs(closures) / source
    closures_config = load_study(folder / "config.toml")
    settings = read_length_settings(closures_config.options)
    if invocation.dry_run:
        logger.info("dry run: %s, scores of %s, %s", config.question, folder, settings)
        return
    verdict = pd.read_csv(closures / "results" / source / "tables" / "verdict.csv")
    families = dict(zip(verdict["model"], verdict["family"], strict=False))
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir,
        invocation.run_id,
        config,
        data_dir=LAB_DIR / "data",
        placement=placement,
    ) as run:
        shutil.copy(folder / "config.toml", run.folder / "closures-config.toml")
        losses = pd.read_parquet(folder / "metrics.parquet")
        tests = length_tests(losses, families, settings)
        run.write_tables({"length": tests, "length_by_market": length_by_market(losses)})
        run.finish(tests)


if __name__ == "__main__":
    main()
