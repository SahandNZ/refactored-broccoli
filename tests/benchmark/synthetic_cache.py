"""A small synthetic forecast cache that follows the 02-forecasts contract.

Three assets (crypto, forex, stock), three windows (leakage ``l1`` in 2024, development ``d1``
and test ``t1`` in 2025), horizons 1, 4 and 24 (every asset has a bar every hour here, so a
label ends ``h`` hours after its decision: ``label_time = time + h``), and four models:

- ``random-walk`` and ``climatology``: baselines; climatology is the true label distribution.
- ``skilled``: a zero-shot model released 2025-01-01 whose median carries half the label.
- ``sampled``: a zero-shot model with two seeds, released 2025-01-01, forecasting only every
  fourth decision (like Kronos); each seed adds its own constant shift (random noise, no skill).

``cache_folder`` is this cache and ``quarters_cache_folder`` the same cache over the real
windows d1-d4 and t1-t4, for the rules that need four test quarters. The tests of
``tests/benchmark/`` get both fixtures from its ``conftest.py``; a study's tests that read the
cache import them in their own ``conftest.py``. A study that needs more models writes its own
cache with `write_cache` and `normal_forecasts` (06-rq4-leakage, 04-rq2-center-vs-scale).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.benchmark._cache import QUANTILE_COLUMNS, write_forecasts
from src.models import QUANTILE_LEVELS

ASSETS = pd.DataFrame(
    {
        "asset": ["BTCUSDT", "EURUSD", "AAPL"],
        "market": ["crypto", "forex", "stock"],
        "cost_per_side_bps": [5.0, 0.4, 1.0],
    }
)
WINDOWS = {"l1": "2024-03-01", "d1": "2025-03-01", "t1": "2025-09-01"}
QUARTER_WINDOWS = {
    "d1": "2024-11-01", "d2": "2025-02-01", "d3": "2025-05-01", "d4": "2025-08-01",
    "t1": "2025-11-01", "t2": "2026-02-01", "t3": "2026-05-01", "t4": "2026-08-01",
}  # fmt: skip
SEED_SHIFTS = {0: 0.1, 1: -0.3}
HORIZONS = (1, 4, 24)
HOURS_PER_WINDOW = 200
MODELS = pd.DataFrame(
    {
        "model": ["random-walk", "climatology", "skilled", "sampled"],
        "kind": ["baseline", "baseline", "zero-shot", "zero-shot"],
        "release_date": ["", "", "2025-01-01", "2025-01-01"],
        "pretraining_cutoff": ["", "", "", "2024-06-30"],
        "every_hours": [0, 0, 0, 4],
    }
)


def _labels(rng: np.random.Generator, windows: dict[str, str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for window, start in windows.items():
        times = pd.date_range(start, periods=HOURS_PER_WINDOW, freq="1h", tz="UTC")
        grid = pd.MultiIndex.from_product(
            [times, ASSETS["asset"], HORIZONS], names=["time", "asset", "horizon"]
        ).to_frame(index=False)
        grid["window"] = window
        parts.append(grid)
    labels = pd.concat(parts, ignore_index=True)
    labels["label_time"] = labels["time"] + pd.to_timedelta(labels["horizon"], unit="h")
    labels["label"] = rng.standard_normal(len(labels))
    labels["sigma"] = 0.01
    labels["log_return"] = labels["label"] * labels["sigma"] * np.sqrt(labels["horizon"])
    return labels


def normal_forecasts(
    labels: pd.DataFrame, center: np.ndarray, scale: float, seed: int
) -> pd.DataFrame:
    """Return the forecast rows of a normal around `center` with `scale` for every label row."""
    keys = labels[["time", "asset", "horizon"]].reset_index(drop=True)
    values = center[:, None] + scale * norm.ppf(np.asarray(QUANTILE_LEVELS))[None, :]
    quantiles = pd.DataFrame(values, columns=list(QUANTILE_COLUMNS))
    return pd.concat([keys.assign(seed=seed), quantiles], axis=1)


def write_cache(folder: Path, windows: dict[str, str]) -> Path:
    """Write the cache of the four models over `windows` (name to start day) into `folder`."""
    rng = np.random.default_rng(0)
    labels = _labels(rng, windows)
    folder.mkdir(parents=True)
    labels.to_parquet(folder / "labels.parquet")
    MODELS.to_csv(folder / "models.csv", index=False)
    ASSETS.to_csv(folder / "assets.csv", index=False)
    zero = np.zeros(len(labels))
    write_forecasts(folder, "random-walk", normal_forecasts(labels, zero, 0.5, 0))
    write_forecasts(folder, "climatology", normal_forecasts(labels, zero, 1.0, 0))
    skilled = 0.5 * labels["label"].to_numpy()
    write_forecasts(folder, "skilled", normal_forecasts(labels, skilled, np.sqrt(0.75), 0))
    every_fourth = labels[labels["time"].dt.hour % 4 == 0].reset_index(drop=True)
    shifted = [
        normal_forecasts(every_fourth, np.full(len(every_fourth), shift), 1.0, seed)
        for seed, shift in SEED_SHIFTS.items()
    ]
    write_forecasts(folder, "sampled", pd.concat(shifted))
    return folder


@pytest.fixture(scope="session")
def cache_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the synthetic cache once per test session and return its run folder."""
    return write_cache(tmp_path_factory.mktemp("02-forecasts") / "runs" / "synthetic", WINDOWS)


@pytest.fixture(scope="session")
def quarters_cache_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the synthetic cache over d1-d4 and t1-t4 and return its run folder."""
    folder = tmp_path_factory.mktemp("02-forecasts") / "runs" / "quarters"
    return write_cache(folder, QUARTER_WINDOWS)
