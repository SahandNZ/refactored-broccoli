"""Fixtures of the 07-rq5-off-hours tests: its ``lib`` package and synthetic bars.

``off_hours_market`` and ``off_hours_lab`` build the bars of the study: a closed-market asset
whose weekend gap is a planted multiple of the weekend crypto move, written to disk;
``off_hours_universe`` writes the ``universe.csv`` of a 01-universe run. ``off_hours_fake``
is a fake model that reads the weekend crypto move from its covariate.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from src import LAB_DIR
from src.benchmark import CacheSettings, cache_settings
from src.data import bars_file
from src.enums import Market, Timeframe
from src.experiment import load_study
from src.models import QUANTILE_LEVELS, Forecast, Forecaster

STUDY_DIR = LAB_DIR / "experiments" / "07-rq5-off-hours"
OFF_HOURS_BASKET = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT")
UNIVERSE_RUN = str(
    load_study(STUDY_DIR.parent / "02-forecasts" / "study.toml").options["universe_run"]
)
"""The 01-universe run the 02-forecasts configuration names (the paper's universe)."""
STUDY_UNIVERSE_RUN = str(load_study(STUDY_DIR / "study.toml").options["universe_run"])
"""The 01-universe run the study itself names."""


@pytest.fixture(name="off_hours", scope="session")
def _off_hours(load_study_lib: Callable[[Path], ModuleType]) -> ModuleType:
    return load_study_lib(STUDY_DIR)


@pytest.fixture(name="benchmark", scope="session")
def _benchmark() -> CacheSettings:
    """The benchmark's settings: 02-forecasts' study.toml stands in for one of its runs."""
    return cache_settings(load_study(STUDY_DIR.parent / "02-forecasts" / "study.toml"))


@pytest.fixture(name="settings", scope="session")
def _settings(off_hours: ModuleType, benchmark: CacheSettings) -> Any:
    """The study's `OffHoursSettings`, read from its ``study.toml``, on the development folds.

    The synthetic bars end in 2025-10, so the tests score d1-d4 even with the locked plan's
    test folds. Typed `Any`: tests never import ``lib`` by name, so its record type has no name
    here.
    """
    config = dataclasses.replace(load_study(STUDY_DIR / "study.toml"), folds="development")
    return off_hours.off_hours_settings(config, benchmark)


# --- a closed market whose weekend gap carries the crypto move ----------------------------


def _is_open(hours: pd.DatetimeIndex, reopens: str) -> np.ndarray:
    # a forex week, open from Sunday 22:00 to Friday 21:00 UTC, or a market that reopens on
    # Monday 07:00 UTC (a European index); both close on Friday 21:00
    day, hour = hours.day_of_week, hours.hour
    is_closed = (day == 5) | ((day == 4) & (hour >= 21))
    if reopens == "sunday":
        is_closed |= (day == 6) & (hour < 22)
    else:
        is_closed |= (day == 6) | ((day == 0) & (hour < 7))
    return ~np.asarray(is_closed)


def make_off_hours_market(
    *, beta: float, seed: int, start: str, end: str, reopens: str = "sunday"
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the 1h bars of a closed-market asset and of the crypto basket.

    Every hour the basket assets share one random step; the asset's log price takes its own
    noise while open and ``beta`` times the basket's step plus noise while closed, so its
    weekend gap is ``beta`` times the weekend crypto move plus noise. Bars exist only while
    the asset is open (closed hours are absent); it reopens on Sunday 22:00 or Monday 07:00
    UTC (`reopens`).
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left", name="open_time")
    common = 0.004 * rng.standard_normal(len(hours))
    steps = common[:, None] + 0.002 * rng.standard_normal((len(hours), len(OFF_HOURS_BASKET)))
    crypto_paths = np.log(100.0) + np.vstack([np.zeros(len(OFF_HOURS_BASKET)), np.cumsum(steps, 0)])
    crypto = {
        asset: pd.DataFrame({"close": np.exp(crypto_paths[1:, column])}, index=hours)
        for column, asset in enumerate(OFF_HOURS_BASKET)
    }
    is_open = _is_open(hours, reopens)
    target_steps = np.where(
        is_open, 0.001 * rng.standard_normal(len(hours)),
        beta * steps.mean(axis=1) + 0.0005 * rng.standard_normal(len(hours)),
    )  # fmt: skip
    path = np.concatenate([[0.0], np.cumsum(target_steps)])
    bars = pd.DataFrame({"mid_open": np.exp(path[:-1]), "mid_close": np.exp(path[1:])}, index=hours)
    return bars[is_open], crypto


@pytest.fixture
def off_hours_market() -> Callable[..., tuple[pd.DataFrame, dict[str, pd.DataFrame]]]:
    """`make_off_hours_market`, for the tests of 07-rq5-off-hours."""
    return make_off_hours_market


def write_off_hours_lab(
    data_dir: Path, markets: dict[str, tuple[str, pd.DataFrame]], crypto: dict[str, pd.DataFrame]
) -> Path:
    """Write the 1h bars of the closed-market assets and the basket in the layout of `read_bars`.

    `markets` maps each symbol to its market and its 1h bars.
    """
    frames = {symbol: (Market(market), bars) for symbol, (market, bars) in markets.items()}
    frames |= {symbol: (Market.CRYPTO, bars) for symbol, bars in crypto.items()}
    for symbol, (market, bars) in frames.items():
        path = bars_file(data_dir, market, symbol, Timeframe.H1)
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(path)
    return data_dir


@pytest.fixture
def off_hours_lab() -> Callable[..., Path]:
    """`write_off_hours_lab`, for the tests of 07-rq5-off-hours."""
    return write_off_hours_lab


def write_off_hours_universe(
    lab_dir: Path, rows: dict[str, tuple[str, bool, bool]], *, run: str | None = None
) -> Path:
    """Write ``universe.csv`` of a 01-universe run below `lab_dir`.

    `rows` maps each symbol to its market, whether it is in the universe and whether it
    passed the data filter. Without `run` the same table goes to both the study's run and
    the 02-forecasts run's (`STUDY_UNIVERSE_RUN`, `UNIVERSE_RUN`).
    """
    table = pd.DataFrame(
        [
            {"symbol": symbol, "market": market, "in_universe": kept, "passed_data_filter": passed}
            for symbol, (market, kept, passed) in rows.items()
        ]
    )
    runs = [run] if run else [STUDY_UNIVERSE_RUN, UNIVERSE_RUN]
    runs_dir = lab_dir / "experiments" / "01-universe" / "runs"
    for name in runs:
        (runs_dir / name / "tables").mkdir(parents=True, exist_ok=True)
        table.to_csv(runs_dir / name / "tables" / "universe.csv", index=False)
    return runs_dir


@pytest.fixture
def off_hours_universe() -> Callable[..., Path]:
    """`write_off_hours_universe`, for the tests of 07-rq5-off-hours."""
    return write_off_hours_universe


_OFF_HOURS_Z = ndtri(np.asarray(QUANTILE_LEVELS))


class OffHoursFakeForecaster(Forecaster):
    """Normal quantiles around the last observed value; with covariates it adds a planted
    multiple of the covariate's move since that value. Records every input."""

    def __init__(self, *, beta: float = 0.8, width: float = 0.004) -> None:
        super().__init__("fake", device="cpu")
        self.beta, self.width = beta, width
        self.inputs: list[tuple[np.ndarray, ...]] = []

    def _forecast(self, centre: np.ndarray, horizon: int) -> Forecast:
        quantiles = centre[:, None, None] * np.exp(self.width * _OFF_HOURS_Z)[None, None, :]
        quantiles = np.repeat(quantiles, horizon, axis=1)
        return Forecast(point=quantiles[:, :, 4], quantiles=quantiles)

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        # around the last observed value (a calendar context ends with closed hours, NaN)
        self.inputs.append((context,))
        return self._forecast(context[np.arange(len(context)), _last_observed(context)], horizon)

    def predict_with_covariates(
        self, context: np.ndarray, covariates: np.ndarray, horizon: int
    ) -> Forecast:
        self.inputs.append((context, covariates))
        last = _last_observed(context)
        rows = np.arange(len(context))
        move = covariates[rows, 0, -1] - covariates[rows, 0, last]
        return self._forecast(context[rows, last] * np.exp(self.beta * move), horizon)


def _last_observed(context: np.ndarray) -> np.ndarray:
    # the position of each row's last finite value
    return np.array([np.flatnonzero(np.isfinite(row))[-1] for row in context])


@pytest.fixture
def off_hours_fake() -> type[OffHoursFakeForecaster]:
    """`OffHoursFakeForecaster`, the fake model of 07-rq5-off-hours."""
    return OffHoursFakeForecaster
