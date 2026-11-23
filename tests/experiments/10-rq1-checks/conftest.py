"""Fixtures of the 10-rq1-checks tests: its ``lib`` package.

The cache fixture ``quarters_cache_folder`` comes from ``tests/benchmark/synthetic_cache.py``,
a plugin of ``tests/conftest.py``; its ``sampled`` model decides every fourth hour, like Kronos.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "10-rq1-checks"


@pytest.fixture(name="checks", scope="session")
def _checks(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
