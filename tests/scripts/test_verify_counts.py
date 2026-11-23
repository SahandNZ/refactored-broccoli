"""The check of the paper's counts against the committed tables (scripts/verify_counts.py)."""

import runpy
from typing import Any

import pytest

from src import LAB_DIR

SCRIPT = LAB_DIR / "scripts" / "verify_counts.py"


@pytest.fixture(name="script", scope="module")
def fixture_script() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


def test_every_count_of_the_paper_matches_the_committed_tables(script: dict[str, Any]) -> None:
    counts = script["all_counts"](LAB_DIR / "experiments")
    assert len(counts) == 31
    assert [count.name for count in counts if not count.is_match] == []


def test_a_count_that_differs_is_flagged(script: dict[str, Any]) -> None:
    count = script["Count"]
    assert not count("RQ1", "cells", 119, 118).is_match
    assert count("RQ4", "median", 0.010, 0.0103485).is_match
