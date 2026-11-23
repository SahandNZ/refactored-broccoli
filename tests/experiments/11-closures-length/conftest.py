"""Fixtures of the 11-closures-length tests: its ``lib`` package."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "11-closures-length"


@pytest.fixture(name="length", scope="session")
def _length(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
