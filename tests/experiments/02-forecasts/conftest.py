"""Fixtures of the 02-forecasts tests: its ``lib`` package.

The synthetic lab (three assets, fake models) and the cache built on it (``lab``, ``run``,
``built``) come from ``tests/benchmark/synthetic_lab.py``, a plugin of ``tests/conftest.py``,
so the tests of this study use the same lab as the tests of the cache builder.
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "02-forecasts"


@pytest.fixture(name="forecasts", scope="session")
def _forecasts(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
