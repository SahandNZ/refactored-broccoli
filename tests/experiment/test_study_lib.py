"""The loader of a study's own ``lib`` package (``tests/conftest.py``)."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


def _fake_study(root: Path, paper: str, study: str, value: int) -> Path:
    """Write a study whose ``lib`` has two modules joined by a relative import."""
    folder = root / paper / study
    (folder / "lib").mkdir(parents=True)
    (folder / "lib" / "_value.py").write_text(f"VALUE = {value}\n")
    (folder / "lib" / "__init__.py").write_text("from ._value import VALUE\n")
    return folder


def test_two_studies_load_their_own_lib_side_by_side(
    tmp_path: Path, load_study_lib: Callable[[Path], ModuleType]
):
    first = load_study_lib(_fake_study(tmp_path, "91-paper", "01-first", 1))
    second = load_study_lib(_fake_study(tmp_path, "91-paper", "02-second", 2))
    assert first.VALUE == 1
    assert second.VALUE == 2
    assert first.__name__ == "lib_91_paper_01_first"
    assert second.__name__ == "lib_91_paper_02_second"
    assert load_study_lib(tmp_path / "91-paper" / "01-first") is first


def test_a_study_without_lib_raises(tmp_path: Path, load_study_lib: Callable[[Path], ModuleType]):
    with pytest.raises(FileNotFoundError, match="lib"):
        load_study_lib(tmp_path / "91-paper" / "03-none")


def test_study_lib_as_lib_refuses_a_test_outside_the_study_test_folders(
    request: pytest.FixtureRequest,
):
    # this test lives in tests/experiment/, which mirrors no study
    with pytest.raises(ValueError, match="tests/experiments/<study>/"):
        request.getfixturevalue("study_lib_as_lib")
