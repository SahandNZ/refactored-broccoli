"""Fixtures of the 03-rq1-null tests: its ``lib`` package.

The cache fixture ``quarters_cache_folder`` comes from ``tests/benchmark/synthetic_cache.py``,
a plugin of ``tests/conftest.py``, so the tests of this study use the same synthetic cache as
the shared benchmark tests.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "03-rq1-null"


@pytest.fixture(name="rq1", scope="session")
def _rq1(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
