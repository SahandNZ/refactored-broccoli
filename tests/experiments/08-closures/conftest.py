"""Fixtures of the 08-closures tests: its ``lib`` package."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

STUDY_DIR = LAB_DIR / "experiments" / "08-closures"


@pytest.fixture(name="closures", scope="session")
def _closures(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)
