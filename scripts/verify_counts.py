"""Check the paper's main RQ1-RQ4 counts against the committed aggregate result tables.

Reads only the CSV files under ``experiments/*/results/`` (in git and in the release), so it
needs no market data, no forecast cache and no ``runs/`` folder. Each check recomputes one
count the paper quotes from the table its source comment names and compares it with the
paper's number; the script prints one line per check and exits with 1 if any differs.

Usage: .venv/bin/python scripts/verify_counts.py
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import LAB_DIR

logger = logging.getLogger(__name__)

RQ1 = Path("03-rq1-null/results/test-2026-11-17/tables")
RQ2 = Path("04-rq2-center-vs-scale/results/test-2026-11-17/tables")
RQ3 = Path("05-rq3-net/results/test-2026-11-17/tables")
RQ4 = Path("06-rq4-leakage/results/test-2026-11-18/tables")
"""The committed test-run tables of each research question, relative to ``experiments/``."""


@dataclass(frozen=True)
class Count:
    """One count of the paper and the value recomputed from the tables."""

    question: str
    name: str
    paper: float
    tables: float

    @property
    def is_match(self) -> bool:
        """Whether the recomputed value equals the paper's (counts exactly, DiD to 3 decimals)."""
        return round(self.tables, 3) == self.paper


def rq1_counts(tables: Path) -> list[Count]:
    """Return the RQ1 counts: scored and testable cells, wins, and the power at 80 %."""
    verdict = pd.read_csv(tables / "verdict.csv")
    scored = verdict[verdict["status"] == "scored"]
    main = scored[scored["family"] == "main"]
    mde = pd.read_csv(tables / "mde.csv")
    testable = mde[mde["quarters"] >= 3]
    counts = [
        Count("RQ1", "main-family cells scored", 119, len(main)),
        Count(
            "RQ1", "main-family cells worse than the baseline", 119, (main["skill_high"] < 0).sum()
        ),
        Count("RQ1", "cells scored", 187, len(scored)),
        Count(
            "RQ1", "cells testable (scored, >= 3 quarters)", 134, (scored["quarters"] >= 3).sum()
        ),
        Count("RQ1", "cells that meet the win criterion", 0, scored["beats_baseline"].sum()),
    ]
    power = ((1, 2, 35, 45), (4, 2, 35, 44), (24, 2, 6, 45), (24, 5, 34, 45))
    for horizon, gain, paper_reached, paper_cells in power:
        cells = testable[testable["horizon"] == horizon]
        reached = (cells["is_reached"] & (cells["mde_pct"] <= gain)).sum()
        label = f"{horizon} bar: MDE <= {gain} % at 80 % power"
        counts += [
            Count("RQ1", f"{label} (cells)", paper_reached, reached),
            Count("RQ1", f"{label} (of testable)", paper_cells, len(cells)),
        ]
    return counts


def rq2_counts(tables: Path) -> list[Count]:
    """Return the RQ2 counts: coverage and both scale diagnostics, at 1 bar and overall."""
    verdict = pd.read_csv(tables / "verdict.csv")
    scored = verdict[(verdict["kind"] == "zero-shot") & (verdict["status"] == "scored")]
    both = scored["is_calibrated"] & scored["beats_reference_scale"]
    one_bar = scored["horizon"] == 1
    return [
        Count("RQ2", "1 bar: cells scored", 65, one_bar.sum()),
        Count(
            "RQ2", "1 bar: coverage criterion met", 27, (one_bar & scored["is_calibrated"]).sum()
        ),
        Count("RQ2", "1 bar: coverage and both diagnostics", 25, (one_bar & both).sum()),
        Count("RQ2", "all horizons: coverage and both diagnostics", 57, both.sum()),
        Count("RQ2", "all horizons: cells scored", 194, len(scored)),
    ]


def rq3_counts(tables: Path) -> list[Count]:
    """Return the RQ3 counts: significant net returns, positive means and the delay check."""
    verdict = pd.read_csv(tables / "verdict.csv")
    scored = verdict[(verdict["kind"] == "zero-shot") & (verdict["status"] == "scored")]
    main = scored[scored["family"] == "main"]
    delayed = pd.read_csv(tables / "delayed.csv")
    paired = delayed[delayed["kind"] == "zero-shot"].merge(
        scored, on=["model", "market", "horizon"], suffixes=("_delayed", "")
    )
    return [
        Count("RQ3", "main-family cells scored", 120, len(main)),
        Count(
            "RQ3",
            "main-family cells with significant net return",
            0,
            main["earns_after_costs"].sum(),
        ),
        Count(
            "RQ3",
            "late-family cells with significant net return",
            0,
            scored["earns_after_costs"].sum() - main["earns_after_costs"].sum(),
        ),
        Count("RQ3", "cells scored", 191, len(scored)),
        Count("RQ3", "cells with positive mean net return", 28, (scored["net_mean"] > 0).sum()),
        Count("RQ3", "delay check: cells with a delayed book", 161, len(paired)),
        Count(
            "RQ3",
            "delay check: lower net mean when delayed",
            86,
            (paired["net_mean_delayed"] < paired["net_mean"]).sum(),
        ),
    ]


def rq4_counts(tables: Path) -> list[Count]:
    """Return the RQ4 counts: the main leakage test, the hourly sensitivity and Kronos."""
    verdict = pd.read_csv(tables / "verdict.csv")
    scored = verdict[verdict["status"] == "scored"]
    leakage = scored[scored["test"] == "leakage"]
    hourly = scored[scored["test"] == "hourly"]
    kronos = leakage[leakage["model"].str.startswith("kronos")]
    return [
        Count("RQ4", "main test: cells scored", 187, len(leakage)),
        Count("RQ4", "main test: cells that show leakage", 0, leakage["rejects"].sum()),
        Count("RQ4", "hourly sensitivity: cells scored", 158, len(hourly)),
        Count("RQ4", "hourly sensitivity: cells that reject", 10, hourly["rejects"].sum()),
        Count("RQ4", "Kronos cells in the main test", 29, len(kronos)),
        Count("RQ4", "Kronos median DiD", 0.010, kronos["did"].median()),
    ]


def all_counts(experiments: Path) -> list[Count]:
    """Return every count of RQ1-RQ4, read from the tables under `experiments`."""
    return [
        *rq1_counts(experiments / RQ1),
        *rq2_counts(experiments / RQ2),
        *rq3_counts(experiments / RQ3),
        *rq4_counts(experiments / RQ4),
    ]


def main() -> None:
    """Print every check and exit with 1 if any count differs from the paper."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    counts = all_counts(LAB_DIR / "experiments")
    for count in counts:
        logger.info(
            "%s  %-4s %-52s paper %-6g tables %-8g",
            "ok  " if count.is_match else "DIFF",
            count.question,
            count.name,
            count.paper,
            round(count.tables, 4),
        )
    differing = [count for count in counts if not count.is_match]
    logger.info("%d of %d counts match the paper", len(counts) - len(differing), len(counts))
    if differing:
        sys.exit(1)


if __name__ == "__main__":
    main()
