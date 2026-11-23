"""A small synthetic lab and the forecast cache that 02-forecasts builds on it.

The lab is a crypto pair (24/7), a forex pair (closed at weekends) and a US stock from Alpaca
(15m bars, regular session only, one 9-day gap), with a universe that keeps the three of them.
The models are fakes: ``fake`` (median at the last close, deciles spread by
``exp(0.01 * offset * sqrt(step))``), ``broken`` (NaN and unordered quantiles) and
``fake-kronos`` (reads candles, adds seeded noise); `lab_config` is a 02-forecasts
configuration of them.

The fixtures ``lab`` (the lab on disk), ``run`` (one cache of every model on it) and ``built``
(the fakes in place of `build_forecaster`) serve the tests of the cache builder
(``tests/benchmark/``, through its ``conftest.py``) and of the study 02-forecasts (through its
own ``conftest.py``).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.benchmark import _build, build_cache
from src.benchmark._settings import BASELINES
from src.enums import Market, Timeframe
from src.experiment import StudyConfig
from src.models import Forecast, Forecaster, KronosSettings

DATA_START, DATA_END = "2024-10-01", "2025-11-03"
CUT = pd.Timestamp("2025-03-15 12:00", tz="UTC")
STOCK_GAP = ("2025-03-01", "2025-03-10")
FIELDS = ("open", "high", "low", "close")
OFFSETS = np.linspace(-1.0, 1.0, 9)
MODELS = (*BASELINES, "fake", "fake-kronos")
OPTIONS = {
    "lookback": 16, "horizons": [1, 4, 24], "span_bars": 24,
    "train_days": 20, "step_hours": {"development": 1, "test": 1, "leakage": 4},
    "kronos_every_hours": 167, "kronos_paths": 3,
    "kronos_tokenizer": "test/tokenizer", "kronos_temperature": 1.0, "kronos_top_p": 0.9,
    "kronos_tokenizer_revision": "c" * 40,
    "q1_min_share": 0.99, "chunk": 700, "moirai_chunk": 5000,
    "linear_lags": [1, 2, 4], "linear_ridge": 1.0, "crypto_fee": "taker",
    "us_stock_half_spread_bps": 1.0, "spread_window": ["2024-10-01", "2024-11-01"],
    "zero_shot": {"fake": ["fake", "test/fake"], "fake-kronos": ["kronos", "test/kronos"],
                  "broken": ["broken", "test/broken"]},
    "revisions": {"fake": "a" * 40, "fake-kronos": "b" * 40, "broken": "d" * 40},
    "release_dates": {"fake": "2025-01-01", "fake-kronos": "2025-06-30", "broken": "2025-01-01"},
    "weight_dates": {"fake": "2025-01-01", "fake-kronos": "2025-06-30", "broken": "2025-01-01"},
    "pretraining_cutoffs": {"fake-kronos": "2024-06-30"},
}  # fmt: skip


def lab_config(models: tuple[str, ...] = MODELS, **changes: object) -> StudyConfig:
    """Return a 02-forecasts configuration of `models` with `OPTIONS` plus `changes`."""
    return StudyConfig(
        name="02-forecasts", question="?", tier=1, markets=(Market.CRYPTO,),
        timeframes=(Timeframe.H1,), folds="development", models=models, seeds=(0, 1),
        plan_locked=False, device="cpu", estimated_hours=1.0, estimated_cost_usd=0.0,
        options={**OPTIONS, **changes},
    )  # fmt: skip


def _prices(index: pd.DatetimeIndex, seed: int) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.005, len(index))))


def write_lab(
    folder: Path, *, change_after_cut: bool, start: str = DATA_START, end: str = DATA_END
) -> tuple[Path, Path]:
    """Write the lab's bars, manifest and universe into `folder`; return the data folder and the
    universe file. With `change_after_cut` every price from `CUT` on gets its own noise."""
    data = folder / "data"
    hours = pd.date_range(start, end, freq="1h", tz="UTC", name="open_time")
    weekday, hour = hours.day_of_week, hours.hour
    is_open = ~((weekday == 4) & (hour >= 21) | (weekday == 5) | (weekday == 6) & (hour < 21))
    quarters = pd.date_range(start, end, freq="15min", tz="UTC", name="open_time")
    gap = (quarters >= pd.Timestamp(STOCK_GAP[0], tz="UTC")) & (
        quarters < pd.Timestamp(STOCK_GAP[1], tz="UTC")
    )
    quarters = quarters[(quarters.day_of_week < 5) & ~gap]
    frames = {
        ("crypto", "BTCUSDT", "1h"): (hours, list(FIELDS), "volume"),
        ("forex", "EURUSD", "1h"): (hours[is_open], [f"mid_{f}" for f in FIELDS], "bid_volume"),
        ("stock", "AAA", "15m"): (quarters, list(FIELDS), "volume"),
    }
    for seed, ((market, symbol, timeframe), (index, columns, volume)) in enumerate(frames.items()):
        price = _prices(index, seed)
        if change_after_cut:
            later = index >= CUT
            price[later] *= np.exp(np.random.default_rng(99).normal(0, 0.02, later.sum()))
        frame = pd.DataFrame(dict.fromkeys(columns, price), index=index).assign(**{volume: 1.0})
        if market == "forex":
            # 0.6 bps inside the spread window (October 2024), 5 bps after it
            frame["spread"] = price * np.where(
                index < pd.Timestamp("2024-11-01", tz="UTC"), 0.6e-4, 5e-4
            )
        path = data / market / symbol / "ohlcv" / f"{timeframe}.pq"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path)
    entries = [
        {"symbol": "BTCUSDT", "market": "crypto", "median_spread_bps": None},
        {"symbol": "EURUSD", "market": "forex", "median_spread_bps": 0.6},
        {"symbol": "AAA", "market": "stock", "median_spread_bps": None},
    ]
    (data / "manifest.json").write_text(json.dumps({"instruments": entries}))
    universe = folder / "universe.csv"
    pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "EURUSD", "AAA", "DROPPED", "OUTSIDE"],
            "market": ["crypto", "forex", "stock", "crypto", "crypto"],
            "in_universe": [True, True, True, False, False],
            "dropped": ["", "", "", "coverage 0.9", ""],
        }
    ).replace("", None).to_csv(universe, index=False)
    return data, universe


class FakeForecaster(Forecaster):
    """Median at the last close, deciles spread by ``exp(0.01 * offset * sqrt(step))``."""

    calls: list[np.ndarray]

    def __init__(self, name: str, *, device: str) -> None:
        super().__init__(name, device=device)
        self.calls = []

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        self.calls.append(context)
        steps = np.sqrt(np.arange(1, horizon + 1))[None, :, None]
        quantiles = context[:, -1, None, None] * np.exp(0.01 * OFFSETS * steps)
        return Forecast(point=quantiles[..., 4], quantiles=quantiles)


class BrokenForecaster(FakeForecaster):
    """Every third series has NaN quantiles and every fifth unordered ones."""

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        quantiles = super().predict(context, horizon).quantiles.copy()
        quantiles[::3] = np.nan
        quantiles[1::5] = quantiles[1::5, :, ::-1]
        return Forecast(point=quantiles[..., 4], quantiles=quantiles)


class FakeKronos(FakeForecaster):
    """Reads candles and adds seeded noise, like a model that samples paths."""

    def __init__(self, name: str, *, device: str, settings: KronosSettings) -> None:
        super().__init__(name, device=device)
        self.settings = settings
        self.candles: list[pd.DataFrame] = []

    def sample_closes(self, candles: list[pd.DataFrame], horizon: int) -> np.ndarray:
        # the fake's nine deciles, each path scaled by its own seeded noise: shape
        # (n_series, n_samples, horizon), paths = deciles cycled to n_samples
        self.candles += candles
        context = np.stack([frame["close"].to_numpy() for frame in candles])
        quantiles = super().predict(context, horizon).quantiles
        levels = np.arange(self.settings.n_samples) % quantiles.shape[-1]
        paths = np.moveaxis(quantiles[..., levels], -1, 1)
        noise = torch.rand(len(candles), self.settings.n_samples).numpy()[:, :, None]
        return paths * (1 + 1e-3 * noise)


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Forecaster]]:
    """Replace `build_forecaster` by the fakes and record every model it builds."""
    made: dict[str, list[Forecaster]] = {}

    def build(kind: str, *, name: str, device: str, **extra: object) -> Forecaster:
        classes = {"broken": BrokenForecaster, "kronos": FakeKronos}
        kind = kind if kind in classes else "fake"
        classes["fake"] = FakeForecaster
        model = classes[kind](name, device=device, **extra)
        made.setdefault(kind, []).append(model)
        return model

    monkeypatch.setattr(_build, "build_forecaster", build)
    monkeypatch.setattr(_build, "require_revision", skip_revision_check)
    return made


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The synthetic lab on disk: its data folder and its universe file."""
    return write_lab(tmp_path_factory.mktemp("lab"), change_after_cut=False)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, lab: tuple[Path, Path]) -> Path:
    """One cache of every model on the synthetic lab."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_build, "build_forecaster", plain_build)
        patch.setattr(_build, "require_revision", skip_revision_check)
        folder = tmp_path_factory.mktemp("run")
        build_cache(folder, lab_config(), data_dir=lab[0], universe=lab[1], models=MODELS)
    return folder


def skip_revision_check(repository: str, revision: str) -> None:
    """Stand in for `src.models.require_revision`: the fakes have no Hugging Face commit."""
    assert len(revision) == 40, (repository, revision)


def plain_build(kind: str, *, name: str, device: str, **extra: object) -> Forecaster:
    """Build a fake in place of `build_forecaster`: `FakeKronos` or `FakeForecaster`."""
    if kind == "kronos":
        return FakeKronos(name, device=device, **extra)
    return FakeForecaster(name, device=device)
