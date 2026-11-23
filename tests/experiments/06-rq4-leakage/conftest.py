"""Fixtures of the 06-rq4-leakage tests: its ``lib`` package and a synthetic cache.

``leakage_cache_folder`` is the cache of ``tests/benchmark/synthetic_cache.py`` over the
leakage quarters l1-l12, the development quarters d1-d4 and the test quarters t1-t4, plus
``leaky``: a zero-shot model released 2025-01-01 with pretraining data up to 2024-06-30,
skilled like ``skilled`` on labels before that cutoff and like ``climatology`` after it. As
in the real cache, the leakage quarters decide every 4 hours (``GRID_HOURS``, 50 times per
quarter) and the other windows every hour (200 times).
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark._cache import write_forecasts
from src.evaluate import LEAKAGE_FOLDS
from tests.benchmark.synthetic_cache import MODELS, QUARTER_WINDOWS, normal_forecasts, write_cache

STUDY_DIR = LAB_DIR / "experiments" / "06-rq4-leakage"
WINDOWS = {
    **{fold.name: str(fold.evaluation_start.date()) for fold in LEAKAGE_FOLDS},
    **QUARTER_WINDOWS,
}
LEAKY_CUTOFF = "2024-06-30"
GRID_HOURS = 4
LEAKAGE = [fold.name for fold in LEAKAGE_FOLDS]


def set_plan(text: str, **settings: object) -> str:
    """Return the study.toml `text` with each ``key = value`` line of `settings` replaced.

    The tests set the plan's values whatever the committed study.toml holds (development or the
    locked test plan); a value is written as TOML (a str, bool, int or list of str).
    """
    for key, value in settings.items():
        literal = str(value).lower() if isinstance(value, bool) else json.dumps(value)
        text, count = re.subn(rf"^{key} = .*$", f"{key} = {literal}", text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"study.toml has {count} lines {key!r}, expected 1")
    return text


@pytest.fixture(name="plan_text", scope="session")
def _plan_text() -> Callable[..., str]:
    return set_plan


@pytest.fixture(name="leakage", scope="session")
def _leakage(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)


@pytest.fixture(scope="session")
def leakage_cache_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the synthetic cache with the model ``leaky`` once and return its run folder."""
    folder = tmp_path_factory.mktemp("02-forecasts") / "runs" / "leakage"
    write_cache(folder, WINDOWS)
    leaky = pd.DataFrame(
        {
            "model": ["leaky"],
            "kind": ["zero-shot"],
            "release_date": ["2025-01-01"],
            "pretraining_cutoff": [LEAKY_CUTOFF],
            "every_hours": [0],
        }
    )
    pd.concat([MODELS, leaky]).to_csv(folder / "models.csv", index=False)
    labels = pd.read_parquet(folder / "labels.parquet")
    hour = pd.DatetimeIndex(labels["time"]).hour.to_numpy()
    is_off_grid = labels["window"].isin(LEAKAGE).to_numpy() & (hour % GRID_HOURS != 0)
    labels = labels.drop(index=labels.index[is_off_grid]).reset_index(drop=True)
    labels.to_parquet(folder / "labels.parquet")
    is_seen = labels["label_time"] < pd.Timestamp(LEAKY_CUTOFF, tz="UTC")
    seen = labels[is_seen].reset_index(drop=True)
    unseen = labels[~is_seen].reset_index(drop=True)
    parts = [
        normal_forecasts(seen, 0.5 * seen["label"].to_numpy(), np.sqrt(0.75), 0),
        normal_forecasts(unseen, np.zeros(len(unseen)), 1.0, 0),
    ]
    write_forecasts(folder, "leaky", pd.concat(parts))
    return folder
