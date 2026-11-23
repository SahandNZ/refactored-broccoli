"""Run 01-universe: select the assets and measure the diversity of each universe (Q1).

Usage: .venv/bin/python experiments/01-universe/run.py [--run-id ID] [--dry-run]
"""

import logging
import shutil
from pathlib import Path

from lib import read_settings, select_universe
from src import LAB_DIR
from src.data import load_data_config
from src.experiment import Placement, parse_invocation, start_run

logger = logging.getLogger("01-universe")


def main() -> None:
    """Select the universe on data before the development folds and save every table."""
    invocation = parse_invocation(__doc__, default_study_dir=Path(__file__).parent)
    config = invocation.config
    settings = read_settings(config.options)
    data = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=LAB_DIR / "data")
    if invocation.dry_run:
        logger.info("dry run: %s, %s, quality limits %s", config.question, settings, data.quality)
        return
    placement = Placement("local", None, None, 1)
    with start_run(
        invocation.study_dir, invocation.run_id, config, data_dir=data.data_dir, placement=placement
    ) as run:
        # the manifest that lists the candidates is frozen with the run
        shutil.copy(data.data_dir / "manifest.json", run.folder / "manifest.json")
        outputs = select_universe(data, settings, seed=config.seeds[0])
        outputs["universe"].to_csv(run.folder / "tables" / "universe.csv", index=False)
        outputs["diversity"].to_csv(run.folder / "tables" / "diversity.csv", index=False)
        outputs["sector_mix"].to_csv(run.folder / "tables" / "sector_mix.csv")
        members = outputs["comparison_universes"]
        members.to_csv(run.folder / "tables" / "comparison_universes.csv", index=False)
        outputs["correlation"].to_parquet(run.folder / "correlation.parquet")
        outputs["holdout_correlation"].to_parquet(run.folder / "holdout_correlation.parquet")
        run.finish(outputs["diversity"])


if __name__ == "__main__":
    main()
