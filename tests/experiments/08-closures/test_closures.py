"""08-closures: forecasts in every arm, the tests and the table, and run.py / analyze.py end to end.

A synthetic lab holds a crypto pair (24/7), a forex pair (closed at weekends) and a US stock
(15m bars, regular session only) from 2024-10 to 2025-11; a fake 02-forecasts run gives the
benchmark's settings. The fake models read their context:

- ``spread``: median at the last value, deciles from the spread of the context's returns
  (NaN ignored); released 2025-05-01.
- ``nan-hurt``: the same, with its median five spreads too high when its context has a NaN.
- ``no-nan``: the same, but raises when it gets a NaN; `missing_value_reason` is patched to name
  a reason for it, so its ``missing`` arm is not applicable.
"""

import json
import math
import re
import runpy
import shutil
import tomllib
import zlib
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from arch.bootstrap import optimal_block_length

import src
from src import LAB_DIR
from src.benchmark import RELEASED_AFTER, benchmark_settings, cache_settings
from src.benchmark._baselines import NORMAL_QUANTILES
from src.enums import Market, Timeframe
from src.experiment import StudyConfig, load_study, read_answers
from src.models import Forecast, Forecaster
from src.panel import hourly_closes, load_panel_bars
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    diebold_mariano,
    has_enough_blocks,
)

STUDY = LAB_DIR / "experiments" / "08-closures"
HOUR = pd.Timedelta(hours=1)
MARKETS = {"BTCUSDT": Market.CRYPTO, "EURUSD": Market.FX, "AAA": Market.STOCK}
MODELS = ("spread", "nan-hurt", "no-nan", "ewma-normal")
SPREAD_RELEASE = pd.Timestamp("2025-05-01", tz="UTC")
NO_NAN_REASON = "the fake raises on NaN"
TOO_FEW_BLOCKS = f"not tested: fewer than {MIN_BOOTSTRAP_BLOCKS} bootstrap blocks"

BENCHMARK_TOML = """
[study]
name = "02-forecasts"
question = "?"
[plan]
locked = false
[data]
tier = 1
markets = ["crypto", "forex", "stock"]
timeframes = ["1h"]
[folds]
use = "development"
[models]
names = ["spread", "nan-hurt", "no-nan", "late", "autumn", "autumn-no-nan", "moirai", "toto",
         "kronos-small"]
[seeds]
values = [0, 1, 2]
[compute]
device = "cpu"
estimated_hours = 1.0
estimated_cost_usd = 0.0
[options]
universe_run = "u"
lookback = 96
horizons = [1, 4, 24]
span_bars = 24
train_days = 20
step_hours = { development = 1, test = 1, leakage = 4 }
kronos_every_hours = 167
kronos_paths = 3
kronos_tokenizer = "test/tokenizer"
kronos_tokenizer_revision = "0000000000000000000000000000000000000000"
kronos_temperature = 1.0
kronos_top_p = 0.9
q1_min_share = 0.99
chunk = 700
moirai_chunk = 5000
linear_lags = [1, 2, 4]
linear_ridge = 1.0
crypto_fee = "taker"
us_stock_half_spread_bps = 1.0
spread_window = ["2020-01-01", "2024-01-01"]
[options.zero_shot]
spread = ["fake", "test/spread"]
nan-hurt = ["nan-hurt", "test/nan-hurt"]
no-nan = ["no-nan", "test/no-nan"]
late = ["fake", "test/late"]
autumn = ["fake", "test/autumn"]
autumn-no-nan = ["no-nan", "test/autumn-no-nan"]
moirai = ["moirai2", "test/moirai"]
toto = ["toto2", "test/toto"]
kronos-small = ["kronos", "test/kronos"]
[options.release_dates]
spread = "2025-05-01"
nan-hurt = "2024-01-01"
no-nan = "2024-01-01"
late = "2026-01-01"
autumn = "2025-10-28"
autumn-no-nan = "2025-10-28"
moirai = "2024-01-01"
toto = "2024-01-01"
kronos-small = "2024-01-01"
[options.weight_dates]
spread = "2025-04-01"
nan-hurt = "2024-01-01"
no-nan = "2024-01-01"
late = "2026-01-01"
autumn = "2025-10-28"
autumn-no-nan = "2025-10-28"
moirai = "2024-01-01"
toto = "2024-01-01"
kronos-small = "2024-01-01"
[options.revisions]
spread = "1111111111111111111111111111111111111111"
nan-hurt = "2222222222222222222222222222222222222222"
no-nan = "3333333333333333333333333333333333333333"
late = "4444444444444444444444444444444444444444"
autumn = "6666666666666666666666666666666666666666"
autumn-no-nan = "7777777777777777777777777777777777777777"
moirai = "8888888888888888888888888888888888888888"
toto = "9999999999999999999999999999999999999999"
kronos-small = "5555555555555555555555555555555555555555"
[options.pretraining_cutoffs]
"""


# Fake models and the synthetic lab


class SpreadForecaster(Forecaster):
    """Median at the last finite value, deciles from the spread of the context's log returns."""

    nan_shift = 0.0

    def __init__(self, name: str, *, device: str) -> None:
        super().__init__(name, device=device)
        self.contexts: list[np.ndarray] = []

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        self.contexts.append(context)
        last = pd.DataFrame(context).ffill(axis=1).to_numpy()[:, -1]
        spread = np.nanstd(np.diff(np.log(context), axis=1), axis=1)
        shift = np.where(np.isnan(context).any(axis=1), self.nan_shift * spread, 0.0)
        steps = np.sqrt(np.arange(1, horizon + 1))[None, :, None]
        z = NORMAL_QUANTILES[None, None, :]
        quantiles = last[:, None, None] * np.exp(
            shift[:, None, None] + spread[:, None, None] * steps * z
        )
        return Forecast(point=quantiles[..., 4], quantiles=quantiles)


class NanHurtForecaster(SpreadForecaster):
    """Its median moves up by five spreads when its context has a NaN."""

    nan_shift = 5.0


class NoNanForecaster(SpreadForecaster):
    """Raises on NaN, like a wrapper that cannot take missing values."""

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        if np.isnan(context).any():
            raise ValueError("no-nan got a NaN")
        return super().predict(context, horizon)


def _build(kind: str, *, name: str, device: str) -> Forecaster:
    classes = {"fake": SpreadForecaster, "nan-hurt": NanHurtForecaster, "no-nan": NoNanForecaster}
    return classes[kind](name, device=device)


def _missing_value_reason(kind: str) -> str | None:
    return NO_NAN_REASON if kind == "no-nan" else None


PINNED: list[tuple[str, str]] = []
"""Every (repository, revision) the study checked before building a model."""


def _require_revision(repository: str, revision: str) -> None:
    PINNED.append((repository, revision))


def _patch_models(
    closures_module: ModuleType, patch: pytest.MonkeyPatch, build: Any = _build
) -> None:
    # fake models, the fake table of missing values, and no network for the pinned commits
    patch.setattr(closures_module, "build_forecaster", build)
    patch.setattr(closures_module, "missing_value_reason", _missing_value_reason)
    patch.setattr(closures_module, "require_revision", _require_revision)


def _settings(closures: ModuleType, study: Path, config: StudyConfig) -> Any:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(closures._closures, "missing_value_reason", _missing_value_reason)
        return closures.closure_settings(config, benchmark_settings(study, config))


def _closes(lab: Path, markets: dict[str, Market]) -> dict[str, pd.Series]:
    return hourly_closes(load_panel_bars(lab / "data", markets, (Timeframe.H1,)))


def _write_data(data: Path) -> None:
    hours = pd.date_range("2024-09-01", "2025-11-03", freq="1h", tz="UTC", name="open_time")
    weekday, hour = hours.day_of_week, hours.hour
    is_open = ~((weekday == 4) & (hour >= 21) | (weekday == 5) | (weekday == 6) & (hour < 21))
    quarters = pd.date_range("2024-09-01", "2025-11-03", freq="15min", tz="UTC", name="open_time")
    fields, mid = ["open", "high", "low", "close"], ["mid_open", "mid_high", "mid_low", "mid_close"]
    frames = {
        ("crypto", "BTCUSDT", "1h"): (hours, fields, "volume"),
        ("forex", "EURUSD", "1h"): (hours[is_open], mid, "bid_volume"),
        ("stock", "AAA", "15m"): (quarters[quarters.day_of_week < 5], fields, "volume"),
    }
    for seed, ((market, symbol, timeframe), (index, columns, volume)) in enumerate(frames.items()):
        steps = np.random.default_rng(seed).normal(0, 0.004, len(index))
        price = 100 * np.exp(np.cumsum(steps))
        frame = pd.DataFrame(dict.fromkeys(columns, price), index=index).assign(**{volume: 1.0})
        path = data / market / symbol / "ohlcv" / f"{timeframe}.pq"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path)
    (data / "manifest.json").write_text(json.dumps({"instruments": []}))


def _write_benchmark_run(paper: Path) -> Path:
    folder = paper / "02-forecasts" / "runs" / "synthetic"
    folder.mkdir(parents=True)
    (folder / "config.toml").write_text(BENCHMARK_TOML)
    assets = pd.DataFrame(
        {"asset": list(MARKETS), "market": [market.value for market in MARKETS.values()],
         "cost_per_side_bps": 1.0}
    )  # fmt: skip
    assets.to_csv(folder / "assets.csv", index=False)
    return folder


def _study_toml(models: tuple[str, ...] = MODELS) -> str:
    config = (STUDY / "study.toml").read_text()
    # the synthetic 02 run, the development folds and an open plan, whatever study.toml says
    config = re.sub(r'(?m)^forecast_cache = ".*"$', 'forecast_cache = "synthetic"', config)
    config = re.sub(r'(?m)^use = ".*"$', 'use = "development"', config)
    config = re.sub(r"(?m)^locked = .*$", "locked = false", config)
    config = config.replace('device = "cuda"', 'device = "cpu"')
    # the synthetic lookback is 96 calendar hours: the study's 32 of 512 scaled is 6
    config = config.replace("min_own_closes = 32", "min_own_closes = 6")
    start, end = config.index("[models]"), config.index("[seeds]")
    names = "[models]\nnames = [" + ", ".join(f'"{name}"' for name in models) + "]\n\n"
    return config[:start] + names + config[end:]


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A lab folder with data, a fake 02-forecasts run and the 08-closures study."""
    root = tmp_path_factory.mktemp("lab")
    _write_data(root / "data")
    paper = root / "experiments"
    _write_benchmark_run(paper)
    study = paper / "08-closures"
    study.mkdir(parents=True)
    shutil.copy(STUDY / "README.md", study / "README.md")
    (study / "study.toml").write_text(_study_toml())
    return root


@pytest.fixture(scope="module")
def tables(closures: ModuleType, lab: Path) -> dict[str, pd.DataFrame]:
    """`closures.closure_losses` of every fake model on the synthetic lab."""
    study = lab / "experiments" / "08-closures"
    config = load_study(study / "study.toml")
    markets = closures.benchmark_assets(study, config)
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch)
        return closures.closure_losses(
            _closes(lab, markets), markets, _settings(closures, study, config)
        )


@pytest.fixture(scope="module")
def settings(closures: ModuleType, lab: Path) -> Any:
    """The `ClosureSettings` of the synthetic study."""
    study = lab / "experiments" / "08-closures"
    return _settings(closures, study, load_study(study / "study.toml"))


# Forecasts and scores


def _pooled(summary: pd.DataFrame, closures: ModuleType) -> pd.DataFrame:
    # the primary rows of the summary, indexed by model, group and arm
    rows = summary.loc[summary["split"] == str(closures.ALL)]
    return rows.set_index(["model", "group", "representation"])


def test_a_model_hurt_by_nan_loses_with_the_missing_arm(
    closures: ModuleType, tables: dict[str, pd.DataFrame], settings: Any
):
    summary = tables["summary"]
    for group in (closures.CLOSED, closures.CONTROL):
        rows = summary.loc[
            (summary["model"] == "nan-hurt")
            & (summary["group"] == str(group))
            & (summary["split"] == str(closures.ALL))
            & summary["representation"].isin(list(closures.REPRESENTATIONS))
        ]
        arms = rows["representation"].to_numpy()
        assert arms[np.argmax(rows["crps"].to_numpy(dtype=float))] == closures.MISSING
    verdict = closures.closure_verdict(
        closures.closure_tests(tables["losses"], settings), alpha=0.05
    )
    row = verdict.loc[
        (verdict["model"] == "nan-hurt") & (verdict["alternative"] == closures.MISSING)
    ].iloc[0]
    assert row["status"] == "scored"
    assert row["dm_statistic"] > 0
    assert row["p_value"] > 0.5
    assert not row["is_better"]
    # the whole skill interval of missing is below zero: worse than trading hours only
    assert row["skill_low"] < row["skill"] < row["skill_high"] < 0
    assert row["is_worse"]
    assert row["verdict"] != closures.MISSING


def test_every_arm_of_a_model_scores_exactly_the_same_decisions_and_labels(
    closures: ModuleType, tables: dict[str, pd.DataFrame]
):
    losses = tables["losses"]
    scored = losses.loc[losses["is_scored"]]
    for (model, group), frame in scored.groupby(["model", "group"]):
        keys = {
            arm: frame.loc[frame["representation"] == arm, ["asset", "time", "label"]]
            .sort_values(["asset", "time"])
            .reset_index(drop=True)
            for arm in frame["representation"].unique()
        }
        assert closures.BENCHMARK_INPUT in keys, (model, group)
        first = next(iter(keys.values()))
        assert len(first) > 0, (model, group)
        for table in keys.values():
            pd.testing.assert_frame_equal(table, first)


def test_a_model_without_nan_support_never_gets_a_nan_and_is_reported_not_applicable(
    closures: ModuleType, tables: dict[str, pd.DataFrame]
):
    summary = _pooled(tables["summary"], closures)
    for group in (closures.CLOSED, closures.CONTROL):
        assert (
            summary.loc[("no-nan", group, closures.MISSING), "status"]
            == f"not applicable: {NO_NAN_REASON}"
        )
        for arm in (closures.TRADING_HOURS, closures.FORWARD_FILLED, closures.BENCHMARK_INPUT):
            assert summary.loc[("no-nan", group, arm), "status"] == "scored"
    losses = tables["losses"]
    arms = set(losses.loc[losses["model"] == "no-nan", "representation"])
    assert arms == {closures.TRADING_HOURS, closures.FORWARD_FILLED, closures.BENCHMARK_INPUT}


def test_zero_shot_models_forecast_only_after_their_release(tables: dict[str, pd.DataFrame]):
    losses = tables["losses"]
    spread = losses.loc[losses["model"] == "spread"]
    assert (spread["time"] > SPREAD_RELEASE).all()
    assert set(spread["window"]) == {"d3", "d4"}
    assert set(losses.loc[losses["model"] == "nan-hurt", "window"]) == {"d1", "d2", "d3", "d4"}


def test_a_model_released_after_the_folds_is_not_built_and_every_built_one_is_pinned(
    closures: ModuleType, lab: Path
):
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("late", "nan-hurt"))
    markets = closures.benchmark_assets(study, config)
    built: list[str] = []

    def build(kind: str, *, name: str, device: str) -> Forecaster:
        built.append(name)
        return _build(kind, name=name, device=device)

    PINNED.clear()
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch, build)
        tables = closures.closure_losses(
            _closes(lab, markets), markets, _settings(closures, study, config)
        )
    assert built == ["test/nan-hurt"]
    assert PINNED == [("test/nan-hurt", "2" * 40)]
    assert set(tables["losses"]["model"]) == {"nan-hurt"}
    late = tables["summary"].loc[tables["summary"]["model"] == "late"]
    assert len(late) == 2 * len(closures.ARMS)
    assert set(late["status"]) == {RELEASED_AFTER}


def test_every_model_is_seeded_and_pinned_before_it_is_built(closures: ModuleType, lab: Path):
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("nan-hurt", "ewma-normal"))
    markets = closures.benchmark_assets(study, config)
    calls: list[str] = []

    def build(kind: str, *, name: str, device: str) -> Forecaster:
        calls.append(f"build {name}")
        return _build(kind, name=name, device=device)

    def pin(repository: str, revision: str) -> None:
        calls.append(f"pin {repository}")

    def seed(seed: int) -> None:
        calls.append(f"seed {seed}")

    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch, build)
        patch.setattr(closures._closures, "require_revision", pin)
        patch.setattr(closures._closures, "seed_everything", seed)
        closures.closure_losses(_closes(lab, markets), markets, _settings(closures, study, config))
    # the baseline is seeded too, though it draws nothing
    assert calls == ["seed 0", "pin test/nan-hurt", "build test/nan-hurt", "seed 0"]


def test_a_model_whose_pinned_commit_is_not_main_is_never_built(closures: ModuleType, lab: Path):
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("nan-hurt",))
    markets = closures.benchmark_assets(study, config)

    built: list[str] = []

    def moved(repository: str, revision: str) -> None:
        raise ValueError(f"{repository}: main is at another commit, pinned {revision}")

    def build(kind: str, *, name: str, device: str) -> Forecaster:
        built.append(name)
        return _build(kind, name=name, device=device)

    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch, build)
        patch.setattr(closures._closures, "require_revision", moved)
        with pytest.raises(ValueError, match="main is at another commit"):
            closures.closure_losses(
                _closes(lab, markets), markets, _settings(closures, study, config)
            )
    # the commit is checked before the model is built, not after
    assert built == []


def test_a_group_without_a_decision_after_the_release_is_reported(closures: ModuleType, lab: Path):
    # released on Tuesday 2025-10-28: the closed markets still decide that week, the Monday
    # control not before d4 ends (2025-11-01)
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("autumn", "autumn-no-nan"))
    markets = closures.benchmark_assets(study, config)
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch)
        tables = closures.closure_losses(
            _closes(lab, markets), markets, _settings(closures, study, config)
        )
    summary = tables["summary"]
    control = summary.loc[summary["group"] == str(closures.CONTROL)]
    assert len(control) == 2 * len(closures.ARMS)
    statuses = control.set_index(["model", "representation"])["status"].to_dict()
    no_decision = "not scored: no decision after its scoring start"
    assert {statuses[("autumn", arm)] for arm in closures.ARMS} == {no_decision}
    # a missing arm the model cannot read keeps its reason
    assert statuses[("autumn-no-nan", closures.MISSING)] == f"not applicable: {NO_NAN_REASON}"
    assert statuses[("autumn-no-nan", closures.TRADING_HOURS)] == no_decision
    closed = summary.loc[
        (summary["group"] == str(closures.CLOSED))
        & (summary["split"] == "all")
        & (summary["model"] == "autumn")
    ]
    assert set(closed["status"]) == {"scored"}
    assert set(tables["losses"]["group"]) == {closures.CLOSED}


def test_a_group_without_a_reopening_in_the_folds_is_left_out(closures: ModuleType, lab: Path):
    # crypto data that ends before the scored folds: the control has no decision at all
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("nan-hurt",))
    markets = closures.benchmark_assets(study, config)
    closes = _closes(lab, markets)
    btc = closes["BTCUSDT"]
    closes["BTCUSDT"] = btc.loc[btc.index < pd.Timestamp("2024-10-01", tz="UTC")]
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch)
        tables = closures.closure_losses(closes, markets, _settings(closures, study, config))
    assert set(tables["summary"]["group"]) == {closures.CLOSED}
    assert closures.best_representations(tables["summary"]).empty


def test_decisions_after_a_data_gap_are_dropped_for_too_few_own_closes_and_counted(
    closures: ModuleType, lab: Path, tables: dict[str, pd.DataFrame]
):
    # O1: without a gap the synthetic lab drops nothing; a stock that lacks a week of data
    # reopens on 2025-03-10 with 0, 1 and 2 own closes in its window: 3 decisions dropped
    assert (tables["summary"]["short_contexts"] == 0).all()
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("nan-hurt",))
    markets = closures.benchmark_assets(study, config)
    closes = _closes(lab, markets)
    stock = closes["AAA"]
    is_gap = (stock.index >= pd.Timestamp("2025-03-03", tz="UTC")) & (
        stock.index < pd.Timestamp("2025-03-09", tz="UTC")
    )
    closes["AAA"] = stock.loc[~is_gap]
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch)
        found = closures.closure_losses(closes, markets, _settings(closures, study, config))
    short = found["summary"].groupby("group")["short_contexts"].max().to_dict()
    assert short == {closures.CLOSED: 3, closures.CONTROL: 0}
    losses = found["losses"]
    day = losses.loc[
        (losses["asset"] == "AAA")
        & (losses["time"] >= pd.Timestamp("2025-03-10", tz="UTC"))
        & (losses["time"] < pd.Timestamp("2025-03-11", tz="UTC"))
    ]
    assert day.empty


def test_a_run_where_no_model_has_a_decision_raises(closures: ModuleType, lab: Path):
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("late",))
    markets = closures.benchmark_assets(study, config)
    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch)
        with pytest.raises(ValueError, match="no model has a decision"):
            closures.closure_losses(
                _closes(lab, markets), markets, _settings(closures, study, config)
            )


def test_the_baseline_reads_no_context_and_is_equal_in_every_arm(
    closures: ModuleType, tables: dict[str, pd.DataFrame], settings: Any
):
    summary = tables["summary"]
    baseline = summary.loc[
        (summary["model"] == str(closures.BASELINE)) & (summary["split"] == str(closures.ALL))
    ]
    assert (baseline["status"] == "scored").all()
    assert len(baseline) == 2 * len(closures.ARMS)
    for _, rows in baseline.groupby("group"):
        values = rows["crps"].to_numpy(dtype=float)
        np.testing.assert_allclose(values, values[0])
    assert closures.BASELINE not in set(closures.closure_tests(tables["losses"], settings)["model"])
    assert closures.BASELINE not in set(closures.best_representations(summary)["model"])


def test_the_groups_are_the_closed_markets_and_crypto_with_its_weekends_cut(
    closures: ModuleType, tables: dict[str, pd.DataFrame]
):
    losses = tables["losses"]
    control = losses.loc[losses["group"] == str(closures.CONTROL)]
    assert set(control["asset"]) == {"BTCUSDT"}
    assert set(losses.loc[losses["group"] == str(closures.CLOSED), "asset"]) == {"EURUSD", "AAA"}
    reopenings = pd.DatetimeIndex(control.loc[control["bar"] == 0, "time"])
    assert set(reopenings.day_of_week) == {0}
    assert set(reopenings.hour) == {0}
    np.testing.assert_allclose(control["closed_hours"].to_numpy(dtype=float), 48.0)
    stock = losses.loc[losses["asset"] == "AAA"]
    assert set(stock["bar"]) == {0, 1, 2}
    assert (stock["closed_hours"] >= 12).all()
    assert set(losses["market"]) == {"crypto", "forex", "stock"}


def test_a_us_stock_reopens_with_the_half_bar_decided_at_nine_new_york(
    closures: ModuleType, tables: dict[str, pd.DataFrame]
):
    # D8: the regular session 09:30-16:00 New York on the 1h grid: the reopening bar is the
    # half bar 09:30-10:00, closing at 10:00, decided at 09:00; nights are 17 hours (16:00 to
    # 09:00) and weekends 65 (64 or 66 over a daylight-saving switch)
    losses = tables["losses"]
    stock = losses.loc[(losses["asset"] == "AAA") & (losses["model"] == "ewma-normal")]
    first = stock.loc[stock["bar"] == 0]
    local = pd.DatetimeIndex(first["time"]).tz_convert("America/New_York")
    assert set(zip(local.hour, local.minute, strict=True)) == {(9, 0)}
    labels = pd.DatetimeIndex(first["label_time"]).tz_convert("America/New_York")
    assert set(labels.hour) == {10}
    hours = np.unique(first["closed_hours"].to_numpy(dtype=float))
    np.testing.assert_allclose(hours, [17.0, 64.0, 65.0, 66.0])
    later = pd.DatetimeIndex(stock.loc[stock["bar"] == 1, "time"]).tz_convert("America/New_York")
    assert set(later.hour) == {10}


def test_the_summary_reports_every_arm_pooled_and_by_bar_and_closure_type(
    closures: ModuleType, tables: dict[str, pd.DataFrame]
):
    summary = tables["summary"]
    pooled = summary.loc[summary["split"] == str(closures.ALL)]
    assert len(pooled) == len(MODELS) * 2 * len(closures.ARMS)
    scored = summary.loc[summary["status"] == "scored"]
    no_closed = scored["representation"].isin([closures.TRADING_HOURS, closures.BENCHMARK_INPUT])
    np.testing.assert_allclose(scored.loc[no_closed, "closed_share"].to_numpy(dtype=float), 0.0)
    calendar = scored.loc[~no_closed]
    assert ((calendar["closed_share"] > 0) & (calendar["closed_share"] < 1)).all()
    assert ((scored["coverage"] >= 0) & (scored["coverage"] <= 1)).all()
    assert (scored["invalid"] == 0).all()
    # the bar splits and the closure-type splits each partition the pooled decisions
    for (model, group, arm), rows in scored.groupby(["model", "group", "representation"]):
        split = rows.set_index("split")["decisions"].astype(int)
        bars = [str(name) for name in split.index if str(name).startswith("bar ")]
        kinds = [str(name) for name in split.index if name in ("night", "weekend", "other")]
        assert bars == ["bar 0", "bar 1", "bar 2"]
        assert split[bars].sum() == split[closures.ALL], (model, arm)
        assert split[kinds].sum() == split[closures.ALL], (model, arm)
        if group == closures.CONTROL:
            assert kinds == ["weekend"]
    splits = set(summary.loc[summary["group"] == str(closures.CLOSED), "split"])
    assert {"night", "weekend"} <= splits


def test_closure_types_split_nights_weekends_and_the_rest(closures: ModuleType):
    # 41 hours: a one-day US stock holiday; 47: the forex weekend at the spring DST switch
    hours = pd.Series([12.0, 23.5, 24.0, 41.0, 43.9, 44.0, 47.0, 48.0, 65.5])
    kinds = closures.closure_types(hours)
    assert list(kinds) == ["night", "night", "other", "other", "other"] + ["weekend"] * 4


def test_a_forecast_of_the_wrong_shape_raises(closures: ModuleType, lab: Path):
    class Wrong(SpreadForecaster):
        def predict(self, context: np.ndarray, horizon: int) -> Forecast:
            quantiles = np.ones((len(context), horizon, 5))
            return Forecast(point=quantiles[..., 2], quantiles=quantiles)

    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("nan-hurt",))
    markets = closures.benchmark_assets(study, config)

    def build(kind: str, *, name: str, device: str) -> Forecaster:
        return Wrong(name, device=device)

    with pytest.MonkeyPatch.context() as patch:
        _patch_models(closures._closures, patch, build)
        with pytest.raises(ValueError, match=r"must have shape \(\d+, 1, 9\)"):
            closures.closure_losses(
                _closes(lab, markets), markets, _settings(closures, study, config)
            )


@pytest.mark.parametrize("chunk", [1, 7, 10_000])
def test_every_predict_call_holds_at_most_the_benchmarks_chunk_and_the_result_is_unchanged(
    closures: ModuleType, lab: Path, settings: Any, chunk: int
):
    # contexts go to a model in calls of at most `chunk` (02-forecasts' chunk), and cutting
    # them changes no forecast
    contexts = closures._closures._group_contexts(_closes(lab, {"EURUSD": Market.FX}), settings)
    runs: dict[int, pd.DataFrame] = {}
    sizes: dict[int, list[int]] = {}
    for size in (chunk, 10**9):
        forecaster = SpreadForecaster("spread", device="cpu")
        chunked = replace(settings, chunks={"spread": size})
        runs[size] = closures._closures._model_losses(forecaster, "spread", contexts, chunked)
        sizes[size] = [len(context) for context in forecaster.contexts]
    assert max(sizes[chunk]) <= chunk
    assert sum(sizes[chunk]) == sum(sizes[10**9])
    if chunk < len(contexts.decisions):
        assert len(sizes[chunk]) > len(sizes[10**9])
    if chunk == 7:
        # a trailing call shorter than the chunk is exercised too
        assert any(0 < size < chunk for size in sizes[chunk])
    pd.testing.assert_frame_equal(runs[chunk], runs[10**9])


class PatchForecaster(SpreadForecaster):
    """Like Toto: refuses a context whose length is not a multiple of its patch of 32."""

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        if context.shape[1] % 32:
            raise ValueError(f"can't divide axis of length {context.shape[1]} in chunks of 32")
        return super().predict(context, horizon)


def test_toto_contexts_are_left_padded_with_their_first_value_to_a_multiple_of_32(
    closures: ModuleType, lab: Path, settings: Any
):
    # short trading-hours contexts (e.g. 60 own closes) reach Toto as 64 values, the first 4
    # repeating the oldest close; full contexts go unchanged
    contexts = closures._closures._group_contexts(_closes(lab, {"AAA": Market.STOCK}), settings)
    assert (contexts.own_bars % 32 != 0).any()
    patched = replace(settings, chunks={"toto": 10**9}, length_multiples={"toto": 32})
    forecaster = PatchForecaster("toto", device="cpu")
    losses = closures._closures._model_losses(forecaster, "toto", contexts, patched)
    assert {len(context[0]) % 32 for context in forecaster.contexts} == {0}
    lengths = [int(length) for length in np.unique(contexts.own_bars) if length % 32]
    length = lengths[0]
    pad = -length % 32
    trading = next(context for context in forecaster.contexts if len(context[0]) == length + pad)
    window = contexts.windows[closures.TRADING_HOURS][contexts.own_bars == length][:, -length:]
    assert pad > 0
    np.testing.assert_array_equal(trading[:, pad:], window)
    np.testing.assert_array_equal(trading[:, :pad], np.repeat(window[:, :1], pad, axis=1))
    assert losses["is_valid"].all()
    # a model without a multiple gets the contexts as they are
    plain = SpreadForecaster("spread", device="cpu")
    closures._closures._model_losses(plain, "spread", contexts, settings)
    assert length in {len(context[0]) for context in plain.contexts}


@pytest.mark.parametrize(
    ("value", "text"),
    [(4253.1, "$>$1000\\rlap{\\textsuperscript{b}}"),
     (1000.01, "$>$1000\\rlap{\\textsuperscript{b}}"), (1000.0, "1000.0"),
     (999.94, "999.9"), (-0.04, "0.0"), (-5.0, "$-$5.0"), (27.7, "27.7")],
)  # fmt: skip
def test_a_change_above_1000_percent_prints_as_over_1000_with_note_b(
    closures: ModuleType, value: float, text: str
):
    assert closures._table._percent(value) == text


def test_left_padding_leaves_a_multiple_unchanged(closures: ModuleType):
    windows = np.arange(64.0).reshape(1, 64)
    np.testing.assert_array_equal(closures._closures._left_pad(windows, 32), windows)
    short = np.arange(1.0, 61.0).reshape(1, 60)
    padded = closures._closures._left_pad(short, 32)
    assert padded.shape == (1, 64)
    np.testing.assert_array_equal(padded[0, :5], [1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(closures._closures._left_pad(short, 1), short)


def test_only_toto_needs_a_length_multiple(closures: ModuleType, lab: Path):
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("spread", "moirai", "toto"))
    assert _settings(closures, study, config).length_multiples == {
        "spread": 1, "moirai": 1, "toto": 32,
    }  # fmt: skip


def test_the_chunk_is_the_benchmarks_and_moirais_its_own(closures: ModuleType, lab: Path):
    # 700 and 5000 in the synthetic 02 run (`chunk`, `moirai_chunk`); Moirai runs in a worker
    study = lab / "experiments" / "08-closures"
    config = replace(load_study(study / "study.toml"), models=("spread", "moirai", "ewma-normal"))
    assert _settings(closures, study, config).chunks == {"spread": 700, "moirai": 5000}


class CrossingForecaster(SpreadForecaster):
    """Crosses its 4th and 5th quantile when the context has a flat (forward-filled) stretch."""

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        forecast = super().predict(context, horizon)
        quantiles = forecast.quantiles.copy()
        is_flat = (np.diff(context, axis=1) == 0).any(axis=1)
        quantiles[is_flat, :, 3] = quantiles[is_flat, :, 5] + 1.0
        return Forecast(point=quantiles[..., 4], quantiles=quantiles)


def test_crossed_quantiles_are_invalid_raw_and_valid_sorted(
    closures: ModuleType, lab: Path, settings: Any
):
    # the forward-filled arm crosses everywhere: the primary rule drops every decision from
    # every representation, the sorted scores keep them all; uncrossed arms score the same
    contexts = closures._closures._group_contexts(_closes(lab, {"EURUSD": Market.FX}), settings)
    forecaster = CrossingForecaster("spread", device="cpu")
    losses = closures._closures._model_losses(forecaster, "spread", contexts, settings)
    filled = losses.loc[losses["representation"] == str(closures.FORWARD_FILLED)]
    assert not filled["is_valid"].any()
    assert filled["is_valid_sorted"].all()
    assert not losses["is_scored"].any()
    assert losses["is_scored_sorted"].all()
    trading = losses.loc[losses["representation"] == str(closures.TRADING_HOURS)]
    np.testing.assert_allclose(trading["crps_sorted"].to_numpy(), trading["crps"].to_numpy())
    assert (trading["is_covered_sorted"] == trading["is_covered"]).all()
    # the sorted CRPS of the crossed arm is the CRPS of the sorted quantiles
    assert not np.allclose(filled["crps_sorted"].to_numpy(), filled["crps"].to_numpy())


def test_a_non_finite_quantile_stays_invalid_when_sorted(closures: ModuleType):
    quantiles = np.tile(np.linspace(-1.0, 1.0, 9), (3, 1))
    quantiles[1, 4] = np.nan
    quantiles[2, [3, 5]] = quantiles[2, [5, 3]]
    labels = np.zeros(3)
    raw = closures._closures._scores(quantiles, labels, "")
    ordered = closures._closures._scores(np.sort(quantiles, axis=1), labels, "_sorted")
    assert list(raw["is_valid"]) == [True, False, False]
    assert list(ordered["is_valid_sorted"]) == [True, False, True]


def test_sorted_robustness_sets_the_sorted_skill_next_to_the_primary_one(
    closures: ModuleType, settings: Any
):
    reference = np.linspace(0.5, 0.7, 25)
    arms = {closures.TRADING_HOURS: reference, closures.FORWARD_FILLED: reference + 0.05}
    losses = _day_losses(closures, arms)
    losses = losses.assign(
        crps_sorted=losses["crps"]
        - np.where(losses["representation"] == closures.FORWARD_FILLED, 0.1, 0.0),
        is_scored_sorted=True,
    )
    # the primary rule lost the first five days (crossed forecasts); sorting keeps them
    is_early = losses["closure_day"] < losses["closure_day"].min() + pd.Timedelta(days=5)
    losses["is_scored"] = ~is_early
    released = _released_at(settings, "2024-01-01")
    tests = closures.closure_tests(losses, released)
    table = closures.sorted_robustness(losses, tests).set_index("alternative")
    row = table.loc[closures.FORWARD_FILLED]
    assert (row["decisions"], row["days"]) == (20, 20)
    assert (row["decisions_sorted"], row["days_sorted"]) == (25, 25)
    assert row["skill"] == pytest.approx(1 - (reference[5:] + 0.05).mean() / reference[5:].mean())
    assert row["skill_sorted"] == pytest.approx(1 - (reference - 0.05).mean() / reference.mean())


def test_an_invalid_forecast_removes_a_decision_from_every_representation_only(
    closures: ModuleType, lab: Path, settings: Any
):
    # an invalid missing forecast removes its decision from all three representations (and
    # the benchmark input); an invalid benchmark-input forecast only from the benchmark input
    markets = {"EURUSD": Market.FX}
    contexts = closures._closures._group_contexts(_closes(lab, markets), settings)
    real = closures._closures._label_forecasts
    broken = {closures.MISSING: 0, closures.BENCHMARK_INPUT: 1}

    def label_forecasts(forecaster: Any, found: Any, arm: str, **options: int) -> np.ndarray:
        quantiles = real(forecaster, found, arm, **options)
        if arm in broken:
            quantiles[broken[arm]] = np.nan
        return quantiles

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(closures._closures, "_label_forecasts", label_forecasts)
        losses = closures._closures._model_losses(None, "a", contexts, settings)
        summary = pd.DataFrame(closures._closures._summary(losses, "a", settings))
    first, second = contexts.decisions.iloc[0], contexts.decisions.iloc[1]
    for arm in closures.ARMS:
        rows = losses.loc[losses["representation"] == arm].set_index(["asset", "time"])
        assert not rows.loc[(first["asset"], first["time"]), "is_scored"], arm
        is_second = bool(rows.loc[(second["asset"], second["time"]), "is_scored"])
        assert is_second == (arm != closures.BENCHMARK_INPUT), arm
    pooled = summary.loc[summary["split"] == str(closures.ALL)]
    arms = pooled["representation"].astype(str).to_list()
    invalid = dict(zip(arms, pooled["invalid"].astype(int).to_list(), strict=True))
    decisions = dict(zip(arms, pooled["decisions"].astype(int).to_list(), strict=True))
    assert invalid == {closures.TRADING_HOURS: 0, closures.FORWARD_FILLED: 0,
                       closures.MISSING: 1, closures.BENCHMARK_INPUT: 1}  # fmt: skip
    total = len(contexts.decisions)
    assert decisions[closures.TRADING_HOURS] == total - 1
    # a split whose only decision is invalid elsewhere blames the other representation
    first_bar = contexts.decisions.iloc[0]
    rows = losses.loc[
        (losses["representation"] == closures.TRADING_HOURS)
        & (losses["asset"] == first_bar["asset"])
        & (losses["time"] == first_bar["time"])
    ]
    status = closures._closures._split_row(rows)["status"]
    assert status == "not scored: invalid in another representation"
    assert decisions[closures.BENCHMARK_INPUT] == total - 2


# The series of closure days


def test_closure_day_means_weigh_each_time_then_each_day_equally(closures: ModuleType):
    day, next_day = pd.Timestamp("2025-01-06", tz="UTC"), pd.Timestamp("2025-01-07", tz="UTC")
    rows = [
        (day, day + 9 * HOUR, "A", 1.0), (day, day + 9 * HOUR, "B", 3.0),
        (day, day + 10 * HOUR, "A", 10.0), (next_day, next_day + 9 * HOUR, "A", 4.0),
        # a later target bar of a closure that ended on `day`, decided after midnight
        (day, next_day + HOUR, "C", 6.0),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["closure_day", "time", "asset", "crps"])
    days = closures.closure_day_means(frame, ["crps"])
    assert list(days.index) == [day, next_day]
    np.testing.assert_allclose(days["crps"].to_numpy(), [(2.0 + 10.0 + 6.0) / 3, 4.0])
    assert list(days["times"]) == [3, 1]


# Tests of the representations against trading hours only


def _day_losses(closures: ModuleType, arms: dict[str, np.ndarray]) -> pd.DataFrame:
    # hand-made scored losses of model "a", one decision per closure day and arm
    days = pd.date_range("2025-01-06", periods=len(next(iter(arms.values()))), tz="UTC")
    return pd.DataFrame(
        [
            {"model": "a", "group": closures.CLOSED, "representation": arm, "asset": "EURUSD",
             "closure_day": day, "time": day + 9 * HOUR, "crps": value, "is_scored": True}
            for arm, values in arms.items()
            for day, value in zip(days, values, strict=True)
        ]
    )  # fmt: skip


def _released_at(settings: Any, release: str) -> Any:
    # the settings with one zero-shot model "a" released at `release`
    return replace(settings, models=("a",), release_dates={"a": pd.Timestamp(release, tz="UTC")})


def test_an_alternative_is_tested_one_sided_against_trading_hours_only_on_closure_days(
    closures: ModuleType, settings: Any
):
    rng = np.random.default_rng(1)
    reference = 0.6 + rng.normal(0, 0.05, 60)
    better = reference - 0.03 + rng.normal(0, 0.01, 60)
    worse = reference + 0.03 + rng.normal(0, 0.01, 60)
    arms = {closures.TRADING_HOURS: reference, closures.FORWARD_FILLED: better,
            closures.MISSING: worse, closures.BENCHMARK_INPUT: reference - 0.2}  # fmt: skip
    tests = closures.closure_tests(
        _day_losses(closures, arms), _released_at(settings, "2024-01-01")
    ).set_index("alternative")
    # the benchmark input is descriptive: never tested
    assert list(tests.index) == [closures.FORWARD_FILLED, closures.MISSING]
    assert (tests["family"] == closures.ON_TIME).all()
    assert (tests["quarters"] == 4).all()
    assert (tests["days"] == 60).all()
    for alternative, values in ((closures.FORWARD_FILLED, better), (closures.MISSING, worse)):
        expected = diebold_mariano(values, reference, overlap=0)
        row = tests.loc[alternative]
        assert row["dm_statistic"] == pytest.approx(expected["statistic"])
        assert row["p_value"] == pytest.approx(expected["p_less"])
        assert row["dm_lags"] == expected["lags"]
        assert row["skill"] == pytest.approx(1 - values.mean() / reference.mean())
        assert row["skill_low"] < row["skill"] < row["skill_high"]
    assert tests.loc[closures.FORWARD_FILLED, "p_value"] < 0.001
    assert tests.loc[closures.MISSING, "p_value"] > 0.999
    verdict = closures.closure_verdict(tests.reset_index(), alpha=0.05)
    assert set(verdict["verdict"]) == {closures.FORWARD_FILLED}
    assert list(verdict["is_worse"]) == [False, True]


def test_each_interval_has_the_study_level_draws_and_its_own_stable_stream(
    closures: ModuleType, settings: Any, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict[str, Any]] = []
    real = closures._verdict.skill_interval

    def interval(alternative: np.ndarray, reference: np.ndarray, **options: Any) -> Any:
        calls.append(options)
        return real(alternative, reference, **options)

    monkeypatch.setattr(closures._verdict, "skill_interval", interval)
    rng = np.random.default_rng(9)
    reference = 0.6 + rng.normal(0, 0.05, 40)
    arms = {closures.TRADING_HOURS: reference,
            closures.FORWARD_FILLED: reference + rng.normal(0, 0.01, 40),
            closures.MISSING: reference + rng.normal(0, 0.01, 40)}  # fmt: skip
    losses = _day_losses(closures, arms)
    both = pd.concat([losses, losses.assign(model="b")], ignore_index=True)
    studied = replace(
        settings,
        models=("b", "a"),
        release_dates={model: pd.Timestamp("2024-01-01", tz="UTC") for model in ("a", "b")},
    )
    closures.closure_tests(both, studied)
    assert [call["level"] for call in calls] == [pytest.approx(1 - settings.alpha)] * 4
    assert {call["n_resamples"] for call in calls} == {settings.n_resamples}
    # the stream: the study's seed, the model's name checksum, the alternative's place
    seeds = [call["seed"] for call in calls]
    a, b = zlib.crc32(b"a"), zlib.crc32(b"b")
    assert seeds == [[0, a, 0], [0, a, 1], [0, b, 0], [0, b, 1]]
    # the same model and data give the same interval whichever other models a run holds, in
    # whatever order (a subset run filters and the study may reorder settings.models)
    alone = closures.closure_tests(losses, replace(studied, models=("a",)))
    full = closures.closure_tests(both, studied)
    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True), full.loc[full["model"] == "a"].reset_index(drop=True)
    )


def test_a_model_with_fewer_than_min_closure_days_gets_no_test_and_no_interval(
    closures: ModuleType, settings: Any
):
    days = settings.min_closure_days - 1
    arms = {closures.TRADING_HOURS: np.linspace(0.5, 0.7, days),
            closures.MISSING: np.linspace(0.4, 0.6, days)}  # fmt: skip
    tests = closures.closure_tests(
        _day_losses(closures, arms), _released_at(settings, "2024-01-01")
    )
    assert tests[["p_value", "skill_low", "skill_high", "dm_statistic"]].isna().all().all()
    assert tests.loc[0, "skill"] == pytest.approx(1 - 0.5 / 0.6)
    status = f"not tested: fewer than {settings.min_closure_days} closure days"
    assert tests.loc[0, "status"] == status
    verdict = closures.closure_verdict(tests, alpha=0.05)
    assert set(verdict["verdict"]) == {status}
    assert np.isnan(verdict.loc[0, "adjusted_p"])
    rng = np.random.default_rng(7)
    enough = {arm: np.append(values, values[-1]) + rng.normal(0, 0.01, days + 1)
              for arm, values in arms.items()}  # fmt: skip
    more = closures.closure_tests(
        _day_losses(closures, enough), _released_at(settings, "2024-01-01")
    )
    assert more.loc[0, "status"] == "scored"


def test_a_test_with_too_few_bootstrap_blocks_is_not_tested(closures: ModuleType, settings: Any):
    # 25 closure days with a slowly swinging difference: a Politis-White block of 6 days
    # leaves fewer than MIN_BOOTSTRAP_BLOCKS blocks (the lab's rule), so no test nor interval
    reference = 0.6 + 0.05 * _ar(0.0, 25, 2)
    difference = 0.01 * np.sin(np.arange(25) / 4) + 0.0005 * _ar(0.0, 25, 3)
    arms = {closures.TRADING_HOURS: reference, closures.FORWARD_FILLED: reference + difference}
    tests = closures.closure_tests(
        _day_losses(closures, arms), _released_at(settings, "2024-01-01")
    )
    assert not has_enough_blocks(25, int(tests.loc[0, "block"]))
    assert tests.loc[0, "status"] == TOO_FEW_BLOCKS
    assert tests[["p_value", "skill_low", "skill_high"]].isna().all().all()
    verdict = closures.closure_verdict(tests, alpha=0.05)
    assert verdict.loc[0, "verdict"] == TOO_FEW_BLOCKS


def test_a_model_without_a_scored_closed_decision_is_listed_as_not_tested(
    closures: ModuleType, settings: Any
):
    losses = _day_losses(closures, {closures.TRADING_HOURS: np.full(25, 0.5),
                                    closures.MISSING: np.full(25, 0.4)})  # fmt: skip
    tests = closures.closure_tests(
        losses.assign(is_scored=False), _released_at(settings, "2024-01-01")
    )
    assert list(tests["alternative"]) == [closures.MISSING]
    assert list(tests["status"]) == ["not tested: no scored decision"]
    assert closures.family_answer(closures.closure_verdict(tests, alpha=0.05), "on time") == (
        "inconclusive"
    )


def test_a_model_with_no_tested_alternative_names_every_reason(closures: ModuleType):
    tests = pd.DataFrame(
        {"model": "a", "family": closures.ON_TIME,
         "alternative": [closures.FORWARD_FILLED, closures.MISSING], "skill": 0.01,
         "skill_high": np.nan, "p_value": np.nan,
         "status": [TOO_FEW_BLOCKS,
                    "not tested: the loss difference has no variance"]}
    )  # fmt: skip
    verdict = closures.closure_verdict(tests, alpha=0.05)
    assert set(verdict["verdict"]) == {
        f"{TOO_FEW_BLOCKS}; not tested: the loss difference has no variance"
    }


def test_a_loss_difference_without_variance_is_not_tested(closures: ModuleType, settings: Any):
    reference = np.random.default_rng(2).uniform(0.4, 0.8, 30)
    arms = {closures.TRADING_HOURS: reference, closures.FORWARD_FILLED: reference - 0.01}
    tests = closures.closure_tests(
        _day_losses(closures, arms), _released_at(settings, "2024-01-01")
    )
    assert tests.loc[0, "status"] == "not tested: the loss difference has no variance"
    assert np.isnan(tests.loc[0, "p_value"])
    verdict = closures.closure_verdict(tests, alpha=0.05)
    assert verdict.loc[0, "verdict"] == "not tested: the loss difference has no variance"
    assert not verdict.loc[0, "is_better"]


def test_representations_scored_on_other_closure_days_are_refused(
    closures: ModuleType, settings: Any
):
    reference = np.linspace(0.5, 0.7, 30)
    losses = _day_losses(closures, {closures.TRADING_HOURS: reference,
                                    closures.FORWARD_FILLED: reference - 0.01})  # fmt: skip
    is_last = (losses["representation"] == closures.FORWARD_FILLED) & (
        losses["closure_day"] == losses["closure_day"].max()
    )
    with pytest.raises(ValueError, match="scored on other closure days"):
        closures.closure_tests(losses.loc[~is_last], _released_at(settings, "2024-01-01"))


@pytest.mark.parametrize(
    ("release", "quarters"),
    [("2024-10-31", 4), ("2024-11-01", 4), ("2024-11-25", 3), ("2025-01-31", 3),
     ("2025-02-01", 3), ("2025-07-31 23:00", 1), ("2025-08-01", 1), ("2025-08-02", 0)],
)  # fmt: skip
def test_a_quarter_counts_when_it_lies_wholly_after_the_scoring_start(
    settings: Any, release: str, quarters: int
):
    # as rq1 (`src.benchmark.full_quarters`): a quarter counts when it starts at or after the
    # scoring start; Chronos-Bolt (2024-11-25) has 3 of d1-d4
    assert _released_at(settings, release).post_release_quarters("a") == quarters


def test_the_study_test_keeps_its_size_with_independent_days_and_is_liberal_with_dependent(
    closures: ModuleType, settings: Any
):
    # through closure_tests on closure-day series with equal expected CRPS: independent days
    # reject about alpha of the time; AR(1) days (rho 0.3) reject more, most with few days
    # (README, Plan changes: about 6.5 % at 250 days and 8 % at 40)
    rng = np.random.default_rng(8)
    quick = replace(_released_at(settings, "2024-01-01"), n_resamples=20)

    def rejection_rate(days: int, rho: float, trials: int) -> float:
        rejections = scored = 0
        for _ in range(trials):
            noise = rng.normal(0, 0.02, days)
            difference = np.empty(days)
            difference[0] = noise[0]
            for day in range(1, days):
                difference[day] = rho * difference[day - 1] + noise[day]
            reference = 0.6 + rng.normal(0, 0.05, days)
            arms = {closures.TRADING_HOURS: reference, closures.MISSING: reference + difference}
            tests = closures.closure_tests(_day_losses(closures, arms), quick)
            # the size of the test itself: a trial gated by the 5-block rule has no p-value
            if tests.loc[0, "status"] == "scored":
                scored += 1
                rejections += float(tests.loc[0, "p_value"]) < settings.alpha
        assert scored > 0.9 * trials
        return rejections / scored

    assert rejection_rate(250, 0.0, 600) == pytest.approx(settings.alpha, abs=0.02)
    assert rejection_rate(40, 0.3, 600) > settings.alpha + 0.01


def test_late_models_form_their_own_holm_family(closures: ModuleType, settings: Any):
    # spread is released 2025-05-01: only d3 and d4 start at or after it (2 < 3 quarters), so late
    assert settings.post_release_quarters("spread") == 2
    assert settings.post_release_quarters("nan-hurt") == 4
    tests = pd.DataFrame(
        [
            ("a", closures.ON_TIME, closures.FORWARD_FILLED, 0.02, 0.01),
            ("a", closures.ON_TIME, closures.MISSING, 0.01, 0.02),
            ("b", closures.ON_TIME, closures.FORWARD_FILLED, 0.02, 0.04),
            ("c", closures.LATE, closures.FORWARD_FILLED, 0.03, 0.03),
        ],
        columns=["model", "family", "alternative", "skill", "p_value"],
    ).assign(skill_high=0.05, status="scored")
    verdict = closures.closure_verdict(tests, alpha=0.05)
    # on time: three tests (0.01 x 3, 0.02 x 2, 0.04 x 1); late: one test, unadjusted
    np.testing.assert_allclose(verdict["adjusted_p"].to_numpy(), [0.03, 0.04, 0.04, 0.03])
    by_model = verdict.groupby("model")["verdict"].first().to_dict()
    assert by_model == {"a": closures.FORWARD_FILLED, "b": closures.FORWARD_FILLED,
                        "c": closures.FORWARD_FILLED}  # fmt: skip
    # pooled into one family the late test would need p * 4 <= alpha, and fail
    pooled = closures.closure_verdict(tests.assign(family=closures.ON_TIME), alpha=0.05)
    assert not pooled.loc[3, "is_better"]


def test_the_closure_tests_put_each_synthetic_model_in_its_family(
    closures: ModuleType, tables: dict[str, pd.DataFrame], settings: Any
):
    tests = closures.closure_tests(tables["losses"], settings)
    families = tests.groupby("model")["family"].first().to_dict()
    assert families == {"nan-hurt": closures.ON_TIME, "no-nan": closures.ON_TIME,
                        "spread": closures.LATE}  # fmt: skip
    assert set(tests.loc[tests["model"] == "no-nan", "alternative"]) == {closures.FORWARD_FILLED}
    assert (tests["days"] >= settings.min_closure_days).all()
    assert (tests["status"] == "scored").all()
    assert (tests["block"] >= 1).all()


@pytest.mark.parametrize(
    ("better", "verdict"),
    [
        ((False, False), "trading hours only"),
        ((True, False), "forward-filled"),
        ((False, True), "missing"),
        ((True, True), "missing"),
    ],
)
def test_the_reference_wins_unless_an_alternative_is_better_and_of_two_the_larger_skill(
    closures: ModuleType, better: tuple[bool, bool], verdict: str
):
    tests = pd.DataFrame(
        {"model": "a", "family": closures.ON_TIME,
         "alternative": [closures.FORWARD_FILLED, closures.MISSING], "skill": [0.01, 0.03],
         "skill_high": [0.02, 0.05], "status": "scored",
         "p_value": [0.001 if is_better else 0.4 for is_better in better]}
    )  # fmt: skip
    assert set(closures.closure_verdict(tests, alpha=0.05)["verdict"]) == {verdict}


@pytest.mark.parametrize("untested_first", [True, False])
def test_an_untested_alternative_leaves_the_verdict_to_the_tested_one(
    closures: ModuleType, untested_first: bool
):
    # missing gives exactly the trading-hours forecast (no variance), forward-filled wins
    rows = [
        {"model": "a", "family": closures.ON_TIME, "alternative": closures.FORWARD_FILLED,
         "skill": 0.02, "skill_high": 0.03, "p_value": 0.03, "status": "scored"},
        {"model": "a", "family": closures.ON_TIME, "alternative": closures.MISSING,
         "skill": 0.0, "skill_high": np.nan, "p_value": np.nan,
         "status": "not tested: the loss difference has no variance"},
    ]  # fmt: skip
    tests = pd.DataFrame(rows[::-1] if untested_first else rows)
    verdict = closures.closure_verdict(tests, alpha=0.05)
    # Holm counts only the tested p (0.03 x 1, not x 2 = 0.06 above alpha)
    tested = verdict.loc[verdict["status"] == "scored"]
    assert tested["adjusted_p"].to_list() == [pytest.approx(0.03)]
    assert set(verdict["verdict"]) == {closures.FORWARD_FILLED}
    assert closures.family_answer(verdict, closures.ON_TIME) == "yes"


def test_q1_is_answered_by_the_family_on_time_and_inconclusive_without_a_tested_model(
    closures: ModuleType,
):
    columns = ["model", "family", "status", "verdict"]
    reference, scored = closures.REFERENCE, "scored"
    too_few = "not tested: fewer than 20 closure days"
    on_time, late = closures.ON_TIME, closures.LATE
    # a winner in the late family does not change the answer of the family on time
    rows = [("a", on_time, scored, reference), ("b", late, scored, "missing")]
    table = pd.DataFrame(rows, columns=columns)
    assert closures.family_answer(table, on_time) == "no"
    assert closures.family_answer(table, late) == "yes"
    rows = [("a", on_time, too_few, too_few), ("b", late, scored, reference)]
    table = pd.DataFrame(rows, columns=columns)
    assert closures.family_answer(table, on_time) == "inconclusive"
    assert closures.family_answer(table, late) == "no"
    assert closures.family_answer(table.iloc[:0], on_time) == "inconclusive"


def test_q2_is_inconclusive_without_a_model_scored_in_both_groups(closures: ModuleType):
    best = pd.DataFrame({"model": ["a", "b"], "is_same": [True, False]})
    assert closures.control_answer(best) == "no"
    assert closures.control_answer(best.iloc[[0]]) == "yes"
    assert closures.control_answer(best.iloc[:0]) == "inconclusive"


def test_best_representations_set_each_models_best_on_both_groups_side_by_side(
    closures: ModuleType,
):
    closed, control, baseline = closures.CLOSED, closures.CONTROL, closures.BASELINE
    trading, filled, missing = closures.TRADING_HOURS, closures.FORWARD_FILLED, closures.MISSING
    benchmark = closures.BENCHMARK_INPUT
    rows = [
        ("a", closed, trading, 0.5), ("a", closed, filled, 0.6),
        ("a", closed, missing, 0.7), ("a", control, trading, 0.4),
        ("a", control, filled, 0.5), ("a", control, missing, 0.6),
        ("b", closed, trading, 0.5), ("b", closed, filled, 0.4),
        ("b", control, trading, 0.4), ("b", control, filled, 0.5),
        (baseline, closed, trading, 0.9), (baseline, control, trading, 0.9),
        ("c", closed, trading, 0.5),
        # the benchmark input is lowest but is not a representation of the same hours
        ("a", closed, benchmark, 0.1), ("a", control, benchmark, 0.1),
    ]  # fmt: skip
    summary = pd.DataFrame(rows, columns=["model", "group", "representation", "crps"])
    summary = summary.assign(status="scored", split=closures.ALL)
    # a descriptive split with a lower CRPS never decides
    split = summary.iloc[[0]].assign(split="bar 0", representation=missing, crps=0.0)
    best = closures.best_representations(pd.concat([summary, split], ignore_index=True))
    best = best.set_index("model")
    assert list(best.index) == ["a", "b"]
    assert (best.loc["a", "closed_best"], best.loc["a", "control_best"]) == (trading, trading)
    assert best.loc["a", "is_same"]
    assert (best.loc["b", "closed_best"], best.loc["b", "control_best"]) == (filled, trading)
    assert not best.loc["b", "is_same"]


# Block length and skill interval


def _ar(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n)
    values = np.empty(n)
    values[0] = noise[0]
    for index in range(1, n):
        values[index] = rho * values[index - 1] + noise[index]
    return values


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_the_block_is_politis_white_rounded_up_with_the_labs_overlap_floor(
    closures: ModuleType, rho: float
):
    values = _ar(rho, 250, 4)
    optimal = float(optimal_block_length(values)["circular"].iloc[0])
    assert closures.block_length(values, overlap=0) == max(1, math.ceil(optimal))
    floor = BLOCK_OVERLAP_FLOOR * 40
    assert closures.block_length(values, overlap=40) == max(floor, math.ceil(optimal))
    # at most the series' length (such a test has too few blocks: `closure_tests`)
    assert closures.block_length(values[:30], overlap=40) == 30


def test_a_constant_series_gets_the_floor_block(closures: ModuleType):
    assert closures.block_length(np.zeros(25), overlap=0) == 1
    assert closures.block_length(np.zeros(25), overlap=2) == BLOCK_OVERLAP_FLOOR * 2
    # a series too short for the estimator gets the floor too, never more than its length
    assert closures.block_length(_ar(0.9, 10, 1), overlap=0) == 1
    assert closures.block_length(_ar(0.9, 3, 1), overlap=2) == 3


def test_the_block_grows_with_the_dependence_of_the_days(closures: ModuleType):
    weak = closures.block_length(_ar(0.0, 400, 5), overlap=0)
    strong = closures.block_length(_ar(0.9, 400, 5), overlap=0)
    assert strong > weak


def test_the_skill_interval_resamples_both_series_with_the_same_blocks(closures: ModuleType):
    rng = np.random.default_rng(6)
    reference = 0.6 + rng.normal(0, 0.1, 100)
    alternative = 0.95 * reference
    # paired draws: the ratio of the means is the same in every resample
    low, high = closures.skill_interval(
        alternative, reference, block=5, level=0.95, n_resamples=500, seed=[0]
    )
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.05)
    noisy = alternative + rng.normal(0, 0.05, 100)
    first = closures.skill_interval(
        noisy, reference, block=5, level=0.95, n_resamples=500, seed=[1]
    )
    again = closures.skill_interval(
        noisy, reference, block=5, level=0.95, n_resamples=500, seed=[1]
    )
    assert first == again
    assert first[0] < 1 - noisy.mean() / reference.mean() < first[1]


# The table of the paper


def _table_summary(closures: ModuleType) -> pd.DataFrame:
    trading, filled, missing = closures.TRADING_HOURS, closures.FORWARD_FILLED, closures.MISSING
    baseline = closures.BASELINE
    rows = [
        (baseline, trading, 0.8, "scored"), (baseline, filled, 0.8, "scored"),
        (baseline, missing, 0.8, "scored"),
        ("nan-hurt", trading, 0.5, "scored"), ("nan-hurt", filled, 0.49999, "scored"),
        ("nan-hurt", missing, 0.6, "scored"),
        ("no-nan", trading, 0.4, "scored"), ("no-nan", filled, 0.38, "scored"),
        ("no-nan", missing, np.nan, f"not applicable: {NO_NAN_REASON}"),
        ("spread", trading, 0.45, "scored"), ("spread", filled, 0.5, "scored"),
        ("spread", missing, 0.4, "scored"),
    ]  # fmt: skip
    closed = pd.DataFrame(rows, columns=["model", "representation", "crps", "status"])
    control = closed.assign(group=closures.CONTROL)
    control.loc[control["model"] == "spread", "status"] = RELEASED_AFTER
    frames = [closed.assign(group=closures.CLOSED), control]
    return pd.concat(frames, ignore_index=True).assign(split=closures.ALL)


def test_the_table_body_bolds_each_verdict_and_orders_late_models_last(
    closures: ModuleType, settings: Any, monkeypatch: pytest.MonkeyPatch
):
    for model in ("spread", "nan-hurt", "no-nan"):
        monkeypatch.setitem(closures._table.MODEL_NAMES, model, model.title())
    verdict = pd.DataFrame(
        {"model": ["nan-hurt", "no-nan", "spread"],
         "verdict": [closures.TRADING_HOURS, closures.FORWARD_FILLED, closures.MISSING]}
    )  # fmt: skip
    body = closures.rq5_table_body(_table_summary(closures), verdict, settings, source="r/x.csv")
    lines = body.splitlines()
    assert lines[0] == (
        "% Generated by experiments/08-closures/make_tab_rq5_closures.py "
        "from r/x.csv. Do not edit by hand."
    )
    not_applicable = "n/a\\rlap{\\textsuperscript{a}}"
    assert lines[1:] == [
        "EWMA-normal & 0.800 & same & same & 0.800 & same & same \\\\",
        "\\addlinespace",
        "Nan-Hurt & \\textbf{0.500} & 0.0 & 20.0 & 0.500 & 0.0 & 20.0 \\\\",
        f"No-Nan & 0.400 & \\textbf{{$-$5.0}} & {not_applicable} & 0.400 & $-$5.0 & "
        f"{not_applicable} \\\\",
        "\\midrule",
        "Spread & 0.450 & 11.1 & \\textbf{$-$11.1} & -- & -- & -- \\\\",
    ]


def test_a_model_not_tested_has_no_bold_in_the_table(
    closures: ModuleType, settings: Any, monkeypatch: pytest.MonkeyPatch
):
    for model in ("spread", "nan-hurt", "no-nan"):
        monkeypatch.setitem(closures._table.MODEL_NAMES, model, model.title())
    status = "not tested: fewer than 20 closure days"
    verdict = pd.DataFrame({"model": ["nan-hurt", "no-nan", "spread"], "verdict": status})
    body = closures.rq5_table_body(_table_summary(closures), verdict, settings, source="x")
    assert "\\textbf" not in body
    assert "Nan-Hurt & 0.500 & 0.0 & 20.0" in body


def test_the_table_body_needs_a_paper_name_for_every_model(closures: ModuleType, settings: Any):
    no_verdict = pd.DataFrame({"model": [], "verdict": []})
    with pytest.raises(KeyError, match="no paper name"):
        closures.rq5_table_body(_table_summary(closures), no_verdict, settings, source="x")


def test_the_table_names_every_model_of_the_study(closures: ModuleType):
    assert set(load_study(STUDY / "study.toml").models) <= set(closures._table.MODEL_NAMES)


# Settings and inputs


def test_the_settings_read_the_named_02_forecasts_run(
    closures: ModuleType, lab: Path, settings: Any
):
    study = lab / "experiments" / "08-closures"
    assert closures.benchmark_assets(study, load_study(study / "study.toml")) == MARKETS
    assert settings.lookback == 96
    assert settings.builds["nan-hurt"] == ("nan-hurt", "test/nan-hurt")
    assert settings.revisions["spread"] == "1" * 40
    # the scoring start is the later of the release and the weight upload
    assert settings.release_dates["spread"] == SPREAD_RELEASE
    assert settings.missing_not_applicable == {"no-nan": NO_NAN_REASON}


def test_the_settings_refuse_kronos_unknown_models_and_a_missing_run(
    closures: ModuleType, lab: Path, tmp_path: Path
):
    study = lab / "experiments" / "08-closures"
    config = load_study(study / "study.toml")
    with pytest.raises(ValueError, match="Kronos is not coded"):
        _settings(closures, study, replace(config, models=("spread", "kronos-small")))
    with pytest.raises(ValueError, match="'nope' is neither ewma-normal nor a model of"):
        _settings(closures, study, replace(config, models=("nope",)))
    with pytest.raises(FileNotFoundError, match=r"set options\.forecast_cache"):
        benchmark_settings(tmp_path / "experiments" / "08-closures", config)


def test_closure_settings_need_one_seed_and_every_option(
    closures: ModuleType, lab: Path, settings: Any
):
    study = lab / "experiments" / "08-closures"
    config = load_study(study / "study.toml")
    assert settings.min_closure_hours == pytest.approx(12.0)
    assert settings.reopen_bars == 3
    assert settings.alpha == pytest.approx(0.05)
    assert settings.small_sample_days == 80
    assert settings.min_own_closes == 6
    assert (settings.min_closure_days, settings.min_quarters) == (20, 3)
    assert (settings.n_resamples, settings.bootstrap_seed) == (2000, 0)
    assert [fold.name for fold in settings.scored_folds] == ["d1", "d2", "d3", "d4"]
    with pytest.raises(ValueError, match="one seed"):
        _settings(closures, study, replace(config, seeds=(0, 1)))
    options_needed = ("alpha", "bootstrap_resamples", "min_closure_days", "min_quarters",
                      "small_sample_days", "min_own_closes")  # fmt: skip
    for option in options_needed:
        options = {key: value for key, value in config.options.items() if key != option}
        with pytest.raises(KeyError):
            _settings(closures, study, replace(config, options=options))


def test_every_option_of_the_study_is_read_by_its_code():
    options = tomllib.loads((STUDY / "study.toml").read_text())["options"]
    code = "".join(
        path.read_text()
        for path in (
            STUDY / "run.py",
            STUDY / "analyze.py",
            STUDY / "lib" / "_closures.py",
            LAB_DIR / "src" / "benchmark" / "_cache.py",
            LAB_DIR / "src" / "benchmark" / "_settings.py",
        )
    )
    assert [key for key in options if f'options["{key}"]' not in code] == []


def test_the_study_names_no_kronos_and_a_reason_for_every_model_without_nan(closures: ModuleType):
    config = load_study(STUDY / "study.toml")
    benchmark = cache_settings(load_study(LAB_DIR / "experiments" / "02-forecasts" / "study.toml"))
    kinds = {
        model: benchmark.zero_shot[model][0]
        for model in config.models
        if model != closures.BASELINE
    }
    assert "kronos" not in kinds.values()
    not_applicable = closures.closure_settings(config, benchmark).missing_not_applicable
    assert set(not_applicable) == {"timesfm-2.5", "timesfm-3", "toto-2-22m", "toto-2-313m"}
    assert set(not_applicable) <= set(config.models)


# run.py and analyze.py end to end


def _main(script: str, study: Path, *arguments: str, run_id: str = "synthetic-run") -> None:
    argv = [script, "--study-dir", str(study), "--run-id", run_id, *arguments]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("sys.argv", argv)
        runpy.run_path(str(STUDY / script), run_name="__main__")


def _check_dropped_counts(text: str) -> None:
    # the decisions dropped for a short trading-hours context: every group in Q1, the
    # control in Q2
    q1, q2 = text.split("**Q2**", 1)
    assert "trading-hours context: closed markets: 0; crypto, weekends cut: 0." in q1
    assert "trading-hours context: crypto, weekends cut: 0." in q2
    assert "closed markets: 0" not in q2


def test_run_and_analyze_answer_both_questions(
    study_lib_as_lib: ModuleType, lab: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lab"
    shutil.copytree(lab, root)
    monkeypatch.setattr(src, "LAB_DIR", root)
    # the scripts' `from lib import ...` finds the study's lib (`study_lib_as_lib`)
    _patch_models(study_lib_as_lib._closures, monkeypatch)
    for model in ("spread", "nan-hurt", "no-nan"):
        monkeypatch.setitem(study_lib_as_lib._table.MODEL_NAMES, model, model)
    study = root / "experiments" / "08-closures"
    # a model released after the scored folds is reported, never built nor in the table
    (study / "study.toml").write_text(_study_toml((*MODELS, "late")))
    _main("run.py", study, "--dry-run")
    assert not (study / "runs").exists()
    with pytest.raises(SystemExit, match="drop --last-days"):
        _main("run.py", study, "--last-days", "3")
    with pytest.raises(SystemExit, match="names a model twice"):
        _main("run.py", study, "--models", "nan-hurt", "nan-hurt")
    _main("run.py", study)
    folder = study / "runs" / "synthetic-run"
    assert json.loads((folder / "meta.json").read_text())["status"] == "completed"
    losses = pd.read_parquet(folder / "metrics.parquet")
    assert set(losses["model"]) == set(MODELS)
    _main("analyze.py", study)
    verdict = pd.read_csv(folder / "tables" / "verdict.csv")
    best = pd.read_csv(folder / "tables" / "best.csv")
    assert set(verdict["model"]) == {"spread", "nan-hurt", "no-nan"}
    assert set(best["model"]) == {"spread", "nan-hurt", "no-nan"}
    assert {"skill", "skill_low", "skill_high", "adjusted_p", "family"} <= set(verdict.columns)
    _main("make_tab_rq5_closures.py", study, "--out-dir", str(tmp_path / "generated"))
    body = (tmp_path / "generated" / "rq5_body.tex").read_text()
    script = "experiments/08-closures/make_tab_rq5_closures.py"
    assert body.startswith(f"% Generated by {script} from synthetic-run/tables/summary.csv")
    assert body.index("nan-hurt") < body.index("\\midrule") < body.index("spread")
    assert "late" not in body
    readme = study / "results" / "README.md"
    answers = read_answers(readme)
    on_time = verdict.loc[verdict["family"] == "on time"]
    winners = on_time.loc[on_time["verdict"] != "trading hours only", "model"].nunique()
    assert answers["Q1"] == ("yes" if winners else "no")
    # on the synthetic data no alternative beats trading hours only, and missing hurts nan-hurt
    assert set(verdict["verdict"]) == {"trading hours only"}
    assert answers["Q1"] == "no"
    hurt = verdict.loc[(verdict["model"] == "nan-hurt") & (verdict["alternative"] == "missing")]
    assert hurt["is_worse"].all()
    assert answers["Q2"] == ("yes" if best["is_same"].all() else "no")
    text = readme.read_text()
    assert f"on time family: {answers['Q1']}, {winners} of 2 models (2 tested)" in text
    expected = (
        "Separate verdict, ",
        "Robustness, descriptive and untested: skill with the nine quantiles sorted",
        "Decisions dropped from every arm for fewer than 6 own closes",
        "late / closed markets / trading hours only: not scored: released after",
        "Invalid forecasts on the closed markets, which remove the decision",
        "late family: ",
        "the 1 crypto assets at one weekly time",
    )
    assert [phrase for phrase in expected if phrase not in text] == []
    _check_dropped_counts(text)
    # an untested row prints its status, never a NaN interval or p-value
    assert "[nan" not in text
    assert "p nan" not in text
    for name in ("verdict.csv", "summary.csv", "best.csv"):
        assert (study / "results" / "synthetic-run" / "tables" / name).exists()


def test_a_development_run_of_a_subset_is_analysed_on_that_subset(
    study_lib_as_lib: ModuleType, lab: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lab"
    shutil.copytree(lab, root)
    monkeypatch.setattr(src, "LAB_DIR", root)
    _patch_models(study_lib_as_lib._closures, monkeypatch)
    for model in ("spread", "nan-hurt", "no-nan"):
        monkeypatch.setitem(study_lib_as_lib._table.MODEL_NAMES, model, model)
    study = root / "experiments" / "08-closures"
    # a development run may forecast a subset of the models; its config.toml names them
    _main("run.py", study, "--models", "nan-hurt", run_id="subset")
    models = pd.read_csv(study / "runs" / "subset" / "tables" / "models.csv")
    assert list(models["model"]) == ["nan-hurt", "ewma-normal"]
    subset = pd.read_parquet(study / "runs" / "subset" / "metrics.parquet")
    # the baseline is always added to a subset
    assert set(subset["model"]) == {"nan-hurt", "ewma-normal"}
    # its analysis covers only the models it ran, not every model of study.toml
    # with a higher small-sample limit its limitations name the liberal test
    toml = study / "runs" / "subset" / "config.toml"
    toml.write_text(toml.read_text().replace("small_sample_days = 80", "small_sample_days = 10000"))
    _main("analyze.py", study, run_id="subset")
    subset_text = (study / "results" / "README.md").read_text()
    assert "a model tested on fewer than 10000 closure days" in subset_text
    subset_tables = study / "runs" / "subset" / "tables"
    assert set(pd.read_csv(subset_tables / "verdict.csv")["model"]) == {"nan-hurt"}
    out = ["--out-dir", str(tmp_path / "generated")]
    _main("make_tab_rq5_closures.py", study, *out, run_id="subset")
    subset_body = (tmp_path / "generated" / "rq5_body.tex").read_text()
    assert (study / "runs" / "subset" / "tables" / "robustness.csv").exists()
    # a run written before the sorted scores existed is analysed without the robustness row
    metrics = study / "runs" / "subset" / "metrics.parquet"
    old = pd.read_parquet(metrics)
    old.drop(columns=[name for name in old.columns if name.endswith("_sorted")]).to_parquet(metrics)
    _main("analyze.py", study, run_id="subset")
    assert (
        "Sorted-quantile robustness: not computed in this run"
        in (study / "results" / "README.md").read_text()
    )
    assert "nan-hurt" in subset_body
    assert "spread" not in subset_body


def test_the_test_run_refuses_a_subset_of_the_models(
    study_lib_as_lib: ModuleType, lab: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lab"
    shutil.copytree(lab, root)
    monkeypatch.setattr(src, "LAB_DIR", root)
    study = root / "experiments" / "08-closures"
    config = (study / "study.toml").read_text()
    config = config.replace("locked = false", "locked = true").replace(
        'use = "development"', 'use = "test"'
    )
    (study / "study.toml").write_text(config)
    with pytest.raises(SystemExit, match="test run of 08-closures forecasts every model"):
        _main("run.py", study, "--models", "nan-hurt")
    # every model in another order is the whole study
    monkeypatch.setattr(study_lib_as_lib._closures, "missing_value_reason", _missing_value_reason)
    _main("run.py", study, "--dry-run", "--models", *reversed(MODELS))
    assert not (study / "runs").exists()
