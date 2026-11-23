"""Shared fixtures of the figure tests: settings restored after each test, PDF page size.

Every test runs inside `matplotlib.rc_context`, so a test that calls `use_paper_style` leaves
the settings of the next test unchanged.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib

matplotlib.use("pdf")

import pytest


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    with matplotlib.rc_context():
        yield


def _page_size(path: Path) -> tuple[float, float]:
    box = re.search(rb"/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", path.read_bytes())
    assert box is not None
    return float(box.group(1)) / 72, float(box.group(2)) / 72


@pytest.fixture
def page_size() -> Callable[[Path], tuple[float, float]]:
    """Return a function giving the width and height in inches of a one-page PDF."""
    return _page_size
