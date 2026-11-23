"""Fixtures of the 12-horizon-spans tests: its ``lib`` package."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "12-horizon-spans"


@pytest.fixture(name="spans", scope="session")
def _spans(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
