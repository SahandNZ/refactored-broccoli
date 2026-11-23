"""Write the bodies of the paper's rq3 tables from a run of 05-rq3-net.

Reads the run's ``tables/verdict.csv`` (written by ``analyze.py``) and ``tables/cells.csv``
and its ``config.toml`` (``reference``, ``table_horizon``, ``alpha``),
and writes ``rq3_body.tex`` (the ``tabular`` of ``tables/rq3.tex``, tab:rq3) and the appendix
table in two bodies, ``rq3_appendix_main_body.tex`` (the main family) and
``rq3_appendix_late_body.tex`` (the late-release family and the reference books), to the
paper's ``tables/generated/``.

Usage: .venv/bin/python experiments/05-rq3-net/make_tab_rq3_break_even.py
--run-id ID [--out-dir DIR]
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from lib import BUY_AND_HOLD, LATE, MAIN, appendix_body, break_even_body, family_models
from src import LAB_DIR
from src.benchmark import study_settings
from src.experiment import load_study, study_runs

logger = logging.getLogger("05-rq3-net")

OUT_DIR = LAB_DIR / "paper" / "tables" / "generated"
"""The folder of the paper's generated table bodies."""


def main() -> None:
    """Read the run's tables and write the three table bodies."""
    # not `src.experiment.parse_invocation`: this script needs an existing run (no default
    # run id) and the folder it writes to
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run = arguments.run_id
    folder = study_runs(arguments.study_dir) / run
    config = load_study(folder / "config.toml")
    options: dict[str, Any] = config.options
    tables = folder / "tables"
    verdict = pd.read_csv(tables / "verdict.csv")
    cells = pd.read_csv(tables / "cells.csv")
    reference, horizon = str(options["reference"]), int(options["table_horizon"])
    level = 1 - study_settings(config).alpha
    bodies = {
        "rq3_body.tex": break_even_body(
            verdict, cells, reference=reference, horizon=horizon, run=run
        ),
    }
    # the appendix in two bodies, so each fits a page (one row per model and market): the
    # main family, and the late-release family with the two reference books
    groups = {
        "rq3_appendix_main_body.tex": [family_models(verdict, MAIN)],
        "rq3_appendix_late_body.tex": [family_models(verdict, LATE), [reference, BUY_AND_HOLD]],
    }
    for name, models in groups.items():
        bodies[name] = appendix_body(
            verdict, cells, groups=models, horizon=horizon, level=level, run=run
        )
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        (arguments.out_dir / name).write_text(body)
        logger.info("wrote %s", arguments.out_dir / name)


if __name__ == "__main__":
    main()
