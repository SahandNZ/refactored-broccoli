"""Fixtures of the 05-rq3-net tests: its ``lib`` package.

The cache fixture ``quarters_cache_folder`` comes from ``tests/benchmark/synthetic_cache.py``,
a plugin of ``tests/conftest.py``, so the tests of this study use the same synthetic cache as
the shared benchmark tests.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "05-rq3-net"


@pytest.fixture(name="rq3", scope="session")
def _rq3(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
