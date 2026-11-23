"""Shared test fixtures: loading a study's own code package (``<study>/lib``).

Every study with its own code keeps it in a package named ``lib``. Scripts import it as
``lib`` (Python puts the script's folder first on ``sys.path``); tests load each study's
``lib`` under its own module name with ``load_study_lib``, so the packages of several studies
never clash. A test that runs a study's ``run.py`` or ``analyze.py`` (with `runpy`) asks for
``study_lib_as_lib``, which also registers that study's ``lib`` as ``lib`` for the one test.

The fixtures that several test folders share, the synthetic forecast cache and the synthetic
lab of the benchmark, are defined once in ``tests/benchmark/`` and loaded here as plugins, so
every test sees the same ones and each session fixture is set up once.
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src import LAB_DIR

pytest_plugins = ["tests.benchmark.synthetic_cache", "tests.benchmark.synthetic_lab"]

TESTS_DIR = Path(__file__).parent


def _load_study_lib(study_dir: Path) -> ModuleType:
    """Load ``<study_dir>/lib`` as the module ``lib_<parent>_<study>`` (``-`` written ``_``).

    Raises
    ------
    FileNotFoundError
        If the study has no ``lib/__init__.py``.
    """
    name = f"lib_{study_dir.parent.name}_{study_dir.name}".replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    init = study_dir / "lib" / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(f"{init} does not exist")
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {init}")
    module = importlib.util.module_from_spec(spec)
    # registered before it runs, so its relative imports find the package
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@pytest.fixture(name="load_study_lib", scope="session")
def _load_study_lib_fixture() -> Callable[[Path], ModuleType]:
    """Return the loader of a study's ``lib`` package (``load_study_lib(study_dir)``)."""
    return _load_study_lib


@pytest.fixture(name="study_lib_as_lib")
def _study_lib_as_lib(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """Register the test's study ``lib`` as ``sys.modules["lib"]`` for one test and return it.

    The study is the one the test's folder mirrors: a test in
    ``tests/experiments/<study>/`` belongs to ``experiments/<study>/``. The
    study's scripts then find its ``lib`` with ``from lib import ...``, as when they run from
    their own folder; the entry is removed after the test.

    Raises
    ------
    ValueError
        If the test is not in a folder ``tests/experiments/<study>/``.
    """
    folder = request.path.parent
    if folder.parent != TESTS_DIR / "experiments":
        raise ValueError(f"{request.path} is not in tests/experiments/<study>/")
    module = _load_study_lib(LAB_DIR / "experiments" / folder.name)
    monkeypatch.setitem(sys.modules, "lib", module)
    return module
