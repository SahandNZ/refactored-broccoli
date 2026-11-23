"""Write the body of the paper's table tab:rq5 from a run of 08-closures.

Reads the run's ``tables/summary.csv`` (written by ``run.py``), ``tables/verdict.csv``
(written by ``analyze.py``) and ``config.toml``, and writes ``rq5_body.tex`` (the rows of the
``tabular`` of ``tables/rq5.tex``) to the paper's ``tables/generated/``.

Usage: .venv/bin/python experiments/08-closures/make_tab_rq5_closures.py
--run-id ID [--out-dir DIR]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from lib import analysis_settings, rq5_table_body
from src import LAB_DIR
from src.experiment import load_study, study_runs

logger = logging.getLogger("08-closures")

OUT_DIR = LAB_DIR / "paper" / "tables" / "generated"
"""The folder of the paper's generated table bodies."""


def main() -> None:
    """Read the run's summary and verdict and write the table body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run = arguments.run_id
    folder = study_runs(arguments.study_dir) / run
    summary = pd.read_csv(folder / "tables" / "summary.csv")
    verdict = pd.read_csv(folder / "tables" / "verdict.csv")
    settings = analysis_settings(arguments.study_dir, load_study(folder / "config.toml"), summary)
    source = f"{run}/tables/summary.csv and verdict.csv"
    body = rq5_table_body(summary, verdict, settings, source=source)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.out_dir / "rq5_body.tex"
    path.write_text(body)
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
