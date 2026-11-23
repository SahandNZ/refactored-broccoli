"""Fixtures of the 04-rq2-center-vs-scale tests: its ``lib`` package and a synthetic cache.

``centre_scale_cache_folder`` is the quarters cache of ``tests/benchmark/synthetic_cache.py``
with climatology given a small positive median (the centre reference needs a sign; real
climatology has the median of the training labels), plus three models for rq2:

- ``ewma-normal``: a baseline, a zero-centred normal twice too wide (the scale reference).
- ``centre-only``: zero-shot, released 2025-01-01; its median carries half the label plus
  as much noise (hit rate about 0.75), but its spread is far too narrow (scale 0.3).
- ``scale-only``: zero-shot, released 2025-01-01; the true N(0, 1) spread around a small
  median of pure noise (no sign skill, calibrated).
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark._cache import write_forecasts
from tests.benchmark.synthetic_cache import MODELS, QUARTER_WINDOWS, normal_forecasts, write_cache

STUDY_DIR = LAB_DIR / "experiments" / "04-rq2-center-vs-scale"

CLIMATOLOGY_MEDIAN = 0.01
"""The median of the synthetic climatology: positive, so every hit counts the up moves."""


@pytest.fixture(name="rq2", scope="session")
def _rq2(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)


@pytest.fixture(scope="session")
def centre_scale_cache_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the quarters cache plus the three models of rq2 and return its run folder."""
    folder = tmp_path_factory.mktemp("02-forecasts") / "runs" / "centre-scale"
    write_cache(folder, QUARTER_WINDOWS)
    added = pd.DataFrame(
        {
            "model": ["ewma-normal", "centre-only", "scale-only"],
            "kind": ["baseline", "zero-shot", "zero-shot"],
            "release_date": ["", "2025-01-01", "2025-01-01"],
            "pretraining_cutoff": ["", "", ""],
            "every_hours": [0, 0, 0],
        }
    )
    pd.concat([MODELS, added]).to_csv(folder / "models.csv", index=False)
    labels = pd.read_parquet(folder / "labels.parquet")
    rng = np.random.default_rng(1)
    label = labels["label"].to_numpy()
    centre = 0.5 * label + 0.5 * rng.standard_normal(len(labels))
    zero = np.zeros(len(labels))
    climatology = np.full(len(labels), CLIMATOLOGY_MEDIAN)
    write_forecasts(folder, "climatology", normal_forecasts(labels, climatology, 1.0, 0))
    write_forecasts(folder, "ewma-normal", normal_forecasts(labels, zero, 2.0, 0))
    write_forecasts(folder, "centre-only", normal_forecasts(labels, centre, 0.3, 0))
    noise = 0.05 * rng.standard_normal(len(labels))
    write_forecasts(folder, "scale-only", normal_forecasts(labels, noise, 1.0, 0))
    return folder
