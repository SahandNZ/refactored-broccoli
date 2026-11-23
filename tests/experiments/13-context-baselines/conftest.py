"""Fixtures of the 13-context-baselines tests: its ``lib`` package."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "13-context-baselines"


@pytest.fixture(name="context", scope="session")
def _context(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
