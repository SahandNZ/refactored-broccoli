"""07-rq5-off-hours: its settings, its asset sets and the weekends of the closed markets."""

import dataclasses
import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.benchmark import CacheSettings
from src.enums import Market, Timeframe
from src.experiment import load_study
from src.panel import ewma_volatility, hourly_closes
from src.utils import week_starts

STUDY = LAB_DIR / "experiments" / "07-rq5-off-hours"
START, END = "2022-01-01", "2024-01-01"


def _crypto(crypto_bars: dict[str, pd.DataFrame]):
    closes = hourly_closes({asset: {Timeframe.H1: bars} for asset, bars in crypto_bars.items()})
    return pd.DataFrame({asset: np.log(series) for asset, series in closes.items()})


# --- weekends -----------------------------------------------------------------------------


def test_every_weekend_runs_from_the_friday_close_to_the_sunday_reopening(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=0, start=START, end=END)
    crypto = _crypto(crypto_bars)
    rows = off_hours.weekend_rows(bars, crypto, settings)
    assert len(rows) == 104
    assert (rows["close_time"].dt.day_of_week == 4).all()
    assert (rows["close_time"].dt.hour == 21).all()
    assert (rows["reopen_time"].dt.day_of_week == 6).all()
    assert (rows["reopen_time"].dt.hour == 22).all()
    assert (rows["label_time"] - rows["reopen_time"] == pd.Timedelta(hours=1)).all()
    assert (rows["week"] == rows["close_time"].dt.normalize() - pd.Timedelta(days=4)).all()
    # the crypto move covers exactly the closed hours, Friday's information the hours before
    row = rows.iloc[10]
    friday_start = row["close_time"] - pd.Timedelta(hours=settings.friday_hours)
    logs = crypto.reindex(pd.DatetimeIndex([friday_start, row["close_time"], row["reopen_time"]]))
    values: np.ndarray = logs.to_numpy(dtype=float)
    np.testing.assert_allclose(row["weekend_crypto"], float((values[2] - values[1]).mean()))
    np.testing.assert_allclose(row["friday_crypto"], float((values[1] - values[0]).mean()))
    open_times = pd.DatetimeIndex(bars.index)
    first = bars.iloc[open_times.get_loc(row["reopen_time"])]
    last = bars.iloc[open_times.get_loc(row["close_time"] - pd.Timedelta(hours=1))]
    assert row["friday_close"] == pytest.approx(last["mid_close"])
    assert row["gap"] == pytest.approx(np.log(first["mid_open"] / last["mid_close"]))
    assert row["move"] == pytest.approx(np.log(first["mid_close"] / last["mid_close"]))


def test_the_label_is_the_move_over_the_benchmarks_ewma_sigma_at_the_friday_close(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=0, start=START, end=END)
    rows = off_hours.weekend_rows(bars, _crypto(crypto_bars), settings)
    closes = pd.Series(bars["mid_close"].to_numpy(), index=bars.index + pd.Timedelta(hours=1))
    sigma = ewma_volatility(closes, pd.DatetimeIndex(rows["close_time"]), span_bars=96)
    assert settings.span_bars == 96
    np.testing.assert_allclose(rows["sigma"], sigma)
    np.testing.assert_allclose(rows["label"], rows["move"] / sigma)
    # the sigma of a weekend ignores every bar after its Friday close
    later = bars.copy()
    later.loc[later.index >= rows["close_time"].iloc[30], "mid_close"] *= 1.3
    changed = off_hours.weekend_rows(later, _crypto(crypto_bars), settings)
    np.testing.assert_array_equal(changed["sigma"][:31], rows["sigma"][:31])


def test_staggered_reopenings_of_one_weekend_share_one_week(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    sunday, crypto_bars = off_hours_market(beta=0.8, seed=0, start=START, end=END)
    monday, _ = off_hours_market(beta=0.8, seed=1, start=START, end=END, reopens="monday")
    crypto = _crypto(crypto_bars)
    early = off_hours.weekend_rows(sunday, crypto, settings)
    late = off_hours.weekend_rows(monday, crypto, settings)
    both = early.merge(late, on="close_time", suffixes=("_sunday", "_monday"))
    assert len(both) == len(late) == 103
    assert (both["reopen_time_monday"].dt.day_of_week == 0).all()
    # Sunday 22:00 and Monday 07:00 fall in different ISO weeks; the Friday close does not
    sunday_weeks = week_starts(pd.DatetimeIndex(both["reopen_time_sunday"]))
    monday_weeks = week_starts(pd.DatetimeIndex(both["reopen_time_monday"]))
    assert (sunday_weeks != monday_weeks).all()
    assert (both["week_sunday"] == both["week_monday"]).all()


def test_a_missing_crypto_bar_leaves_the_basket_move_missing_never_filled(
    off_hours: ModuleType, settings: Any, off_hours_market: Callable[..., Any]
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=0, start=START, end=END)
    crypto = _crypto(crypto_bars)
    rows = off_hours.weekend_rows(bars, crypto, settings)
    crypto.loc[rows["reopen_time"].iloc[0], "SOLUSDT"] = np.nan
    rows = off_hours.weekend_rows(bars, crypto, settings)
    assert np.isnan(rows["weekend_crypto"].iloc[0])
    assert rows.iloc[1:]["weekend_crypto"].notna().all()


def test_basket_return_is_the_mean_log_return_of_the_assets(off_hours: ModuleType):
    times = pd.date_range("2024-01-05 18:00", periods=4, freq="1h", tz="UTC")
    crypto = pd.DataFrame({"A": [0.0, 0.1, 0.2, 0.3], "B": [1.0, 1.0, 1.4, 0.8]}, index=times)
    moves = off_hours.basket_return(crypto, times[[0, 1]], times[[3, 2]])
    np.testing.assert_allclose(moves, [(0.3 - 0.2) / 2, (0.1 + 0.4) / 2])


def test_nothing_after_the_reopening_changes_an_input(
    off_hours: ModuleType,
    settings: Any,
    off_hours_market: Callable[..., Any],
    off_hours_lab: Callable[..., Any],
    tmp_path: Path,
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=1, start=START, end=END)
    lab = off_hours_lab(tmp_path, {"FX": ("forex", bars)}, crypto_bars)
    before = off_hours.read_weekends(lab, {"FX": Market.FX}, settings).table
    cut = before["reopen_time"].iloc[40]
    # scramble the asset after its reopening's first bar and the basket after the reopening
    changed = bars.copy()
    changed.loc[changed.index > cut, ["mid_open", "mid_close"]] *= 1.5
    crypto_changed = {
        asset: frame.assign(close=np.where(frame.index + pd.Timedelta(hours=1) > cut, 2.0, 1.0)
                            * frame["close"])
        for asset, frame in crypto_bars.items()
    }  # fmt: skip
    after_lab = off_hours_lab(tmp_path / "after", {"FX": ("forex", changed)}, crypto_changed)
    after = off_hours.read_weekends(after_lab, {"FX": Market.FX}, settings).table
    inputs = [
        "close_time",
        "reopen_time",
        "sigma",
        "friday_target",
        "friday_crypto",
        "weekend_crypto",
    ]
    pd.testing.assert_frame_equal(before.loc[:40, inputs], after.loc[:40, inputs])
    pd.testing.assert_frame_equal(before.loc[:40, ["gap", "move"]], after.loc[:40, ["gap", "move"]])


def test_read_weekends_keeps_the_window_and_refuses_no_asset(
    off_hours: ModuleType,
    settings: Any,
    off_hours_market: Callable[..., Any],
    off_hours_lab: Callable[..., Any],
    tmp_path: Path,
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=0, start="2019-06-01", end="2025-12-01")
    lab = off_hours_lab(tmp_path, {"FX": ("forex", bars), "GOLD": ("commodity", bars)}, crypto_bars)
    weekends = off_hours.read_weekends(lab, {"FX": Market.FX, "GOLD": Market.COMMODITY}, settings)
    table = weekends.table
    assert set(table["asset"]) == {"FX", "GOLD"}
    assert set(table.loc[table["asset"] == "GOLD", "market"]) == {"commodity"}
    assert (table["reopen_time"] >= settings.start).all()
    assert (table["reopen_time"] < settings.end).all()
    with pytest.raises(ValueError, match="at least one closed-market asset"):
        off_hours.read_weekends(lab, {}, settings)


def _hours(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(list(values), tz="UTC")


def test_a_weekend_reopens_on_sunday_or_monday_and_a_tuesday_reopening_is_not_one(
    off_hours: ModuleType,
):
    # close times of 1h bars: the market closes after Friday 20:00 and reopens on Sunday 22:00
    sunday = _hours("2024-03-08 20:00", "2024-03-08 21:00", "2024-03-10 23:00")
    monday = _hours("2024-03-08 21:00", "2024-03-11 08:00")
    tuesday = _hours("2024-03-29 21:00", "2024-04-02 08:00")
    midweek = _hours("2024-12-24 21:00", "2024-12-26 08:00")
    assert off_hours.weekend_closures(sunday, min_closed_hours=24, max_closed_hours=96)[
        "position"
    ].tolist() == [2]
    assert off_hours.weekend_closures(monday, min_closed_hours=24, max_closed_hours=96)[
        "position"
    ].tolist() == [1]
    assert off_hours.weekend_closures(tuesday, min_closed_hours=24, max_closed_hours=96).empty
    assert off_hours.weekend_closures(midweek, min_closed_hours=24, max_closed_hours=96).empty


# --- settings -----------------------------------------------------------------------------


def test_settings_refuse_a_missing_option_two_seeds_and_no_closed_market(
    off_hours: ModuleType, benchmark: CacheSettings
):
    config = load_study(STUDY / "study.toml")
    options = dict(config.options)
    del options["hac_lags"]
    with pytest.raises(KeyError, match="hac_lags"):
        off_hours.off_hours_settings(dataclasses.replace(config, options=options), benchmark)
    with pytest.raises(ValueError, match="one seed"):
        off_hours.off_hours_settings(dataclasses.replace(config, seeds=(0, 1)), benchmark)
    with pytest.raises(ValueError, match="closed market"):
        off_hours.off_hours_settings(
            dataclasses.replace(config, markets=(Market.CRYPTO,)), benchmark
        )
    with pytest.raises(ValueError, match="not zero-shot models"):
        off_hours.off_hours_settings(dataclasses.replace(config, models=("nope",)), benchmark)


def test_settings_refuse_a_model_that_fills_nan_or_takes_no_covariate(
    off_hours: ModuleType, benchmark: CacheSettings
):
    config = load_study(STUDY / "study.toml")
    with pytest.raises(ValueError, match=r"cannot read NaN as missing.*timesfm-3"):
        off_hours.off_hours_settings(
            dataclasses.replace(config, models=("chronos-2", "timesfm-3")), benchmark
        )
    with pytest.raises(ValueError, match="take no past covariates"):
        off_hours.off_hours_settings(
            dataclasses.replace(config, models=("chronos-bolt-small",)), benchmark
        )


def test_the_models_their_revisions_and_the_context_are_the_benchmarks(
    settings: Any, benchmark: CacheSettings
):
    assert list(settings.zero_shot) == ["chronos-2", "tirex-2", "t0-gamma"]
    assert settings.lookback == benchmark.lookback
    assert settings.span_bars == benchmark.span_bars
    assert settings.zero_shot["chronos-2"] == benchmark.zero_shot["chronos-2"]
    assert settings.revisions == {model: benchmark.revisions[model] for model in settings.zero_shot}


def test_the_scored_folds_follow_the_folds_of_the_configuration(
    off_hours: ModuleType, settings: Any, benchmark: CacheSettings
):
    config = load_study(STUDY / "study.toml")
    test = off_hours.off_hours_settings(dataclasses.replace(config, folds="test"), benchmark)
    assert [fold.name for fold in settings.scored_folds] == ["d1", "d2", "d3", "d4"]
    assert [fold.name for fold in test.scored_folds] == ["t1", "t2", "t3", "t4"]
    assert test.end == pd.Timestamp("2026-11-01", tz="UTC")
    assert settings.width_factors[0] == pytest.approx(0.5)
    assert settings.width_factors[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(np.diff(settings.width_factors), 0.05)
    assert settings.interval_level == pytest.approx(0.95)


def test_every_option_of_rq5_is_read_by_its_code():
    options = tomllib.loads((STUDY / "study.toml").read_text())["options"]
    sources = (
        STUDY / "lib" / "_settings.py",
        LAB_DIR / "src" / "benchmark" / "_cache.py",
        LAB_DIR / "src" / "benchmark" / "_settings.py",
    )
    code = "".join(path.read_text() for path in sources)
    assert [key for key in options if f'options["{key}"]' not in code] == []


# --- the asset sets -----------------------------------------------------------------------


def test_the_asset_sets_come_from_the_studys_universe_run(
    off_hours: ModuleType, settings: Any, off_hours_universe: Callable[..., Any], tmp_path: Path
):
    study = tmp_path / "experiments" / "07-rq5-off-hours"
    study.mkdir(parents=True)
    run = study.parent / "02-forecasts" / "runs" / "synthetic"
    run.mkdir(parents=True)
    shutil.copy(STUDY.parent / "02-forecasts" / "study.toml", run / "config.toml")
    off_hours_universe(
        tmp_path,
        {
            "EURUSD": ("forex", True, True),
            "GBPUSD": ("forex", False, True),
            "WTI": ("commodity", False, True),
            "USATECH": ("index", False, True),
            "ITA40": ("index", False, False),
            "VIX": ("index", False, False),
            "EURDKK": ("forex", False, False),
            "EURMXN": ("forex", False, False),
            "BTCUSDT": ("crypto", True, True),
            "AAPL": ("stock", True, True),
        },
    )
    config = load_study(STUDY / "study.toml")
    config = dataclasses.replace(config, options=config.options | {"forecast_cache": "synthetic"})
    sets = off_hours.universe_assets(study, config, settings)
    assert sets.forecast == {"EURUSD": Market.FX}
    assert sets.gap == {
        "EURUSD": Market.FX,
        "GBPUSD": Market.FX,
        "WTI": Market.COMMODITY,
        "USATECH": Market.INDEX,
    }
    assert sets.universe_run == load_study(STUDY / "study.toml").options["universe_run"]
    assert sets.universe_run != load_study(run / "config.toml").options["universe_run"]


def test_a_second_closure_in_the_same_week_is_not_a_second_weekend(off_hours: ModuleType):
    # a data gap from Sunday 23:00 to Monday 10:00 right after the Sunday reopening
    times = _hours(
        "2024-03-08 21:00", "2024-03-10 23:00", "2024-03-11 11:00", "2024-03-15 21:00",
        "2024-03-17 23:00",
    )  # fmt: skip
    found = off_hours.weekend_closures(times, min_closed_hours=6, max_closed_hours=96)
    assert found["position"].tolist() == [1, 4]


def test_a_closure_longer_than_the_limit_is_not_a_weekend(off_hours: ModuleType):
    # a five-day outage from Wednesday to Monday is not a weekend; a Good Friday one is
    outage = _hours("2024-03-06 21:00", "2024-03-11 08:00")
    good_friday = _hours("2024-03-28 21:00", "2024-03-31 23:00")
    assert off_hours.weekend_closures(outage, min_closed_hours=24, max_closed_hours=96).empty
    found = off_hours.weekend_closures(good_friday, min_closed_hours=24, max_closed_hours=96)
    assert found["position"].tolist() == [1]


def test_the_study_reads_its_own_universe_run_and_refuses_another_universe(
    off_hours: ModuleType, settings: Any, off_hours_universe: Callable[..., Any], tmp_path: Path
):
    # the study's run may differ from the benchmark's pin (e.g. new columns for Q2), but its
    # universe must be the benchmark's
    study = tmp_path / "experiments" / "07-rq5-off-hours"
    study.mkdir(parents=True)
    run = study.parent / "02-forecasts" / "runs" / "synthetic"
    run.mkdir(parents=True)
    shutil.copy(STUDY.parent / "02-forecasts" / "study.toml", run / "config.toml")
    pinned = {"EURUSD": ("forex", True, True), "GBPUSD": ("forex", False, True)}
    off_hours_universe(tmp_path, pinned)
    off_hours_universe(tmp_path, pinned | {"WTI": ("commodity", False, True)}, run="same-universe")
    off_hours_universe(tmp_path, pinned | {"GBPUSD": ("forex", True, True)}, run="other-universe")
    config = load_study(STUDY / "study.toml")
    options = config.options | {"forecast_cache": "synthetic", "universe_run": "same-universe"}
    sets = off_hours.universe_assets(study, dataclasses.replace(config, options=options), settings)
    assert sets.universe_run == "same-universe"
    assert set(sets.gap) == {"EURUSD", "GBPUSD", "WTI"}
    options = options | {"universe_run": "other-universe"}
    with pytest.raises(ValueError, match=r"another universe than the benchmark's.*GBPUSD"):
        off_hours.universe_assets(study, dataclasses.replace(config, options=options), settings)


def test_the_pinned_universe_run_may_lack_the_data_filter_column(
    off_hours: ModuleType, settings: Any, off_hours_universe: Callable[..., Any], tmp_path: Path
):
    # 02's pinned run predates passed_data_filter: only its universe is compared
    study = tmp_path / "experiments" / "07-rq5-off-hours"
    study.mkdir(parents=True)
    run = study.parent / "02-forecasts" / "runs" / "synthetic"
    run.mkdir(parents=True)
    shutil.copy(STUDY.parent / "02-forecasts" / "study.toml", run / "config.toml")
    rows = {"EURUSD": ("forex", True, True), "GBPUSD": ("forex", False, True)}
    off_hours_universe(tmp_path, rows, run="with-column")
    pinned = load_study(run / "config.toml").options["universe_run"]
    folder = off_hours_universe(tmp_path, rows, run=pinned) / pinned / "tables" / "universe.csv"
    pd.read_csv(folder).drop(columns="passed_data_filter").to_csv(folder, index=False)
    config = load_study(STUDY / "study.toml")
    options = config.options | {"forecast_cache": "synthetic", "universe_run": "with-column"}
    sets = off_hours.universe_assets(study, dataclasses.replace(config, options=options), settings)
    assert sets.forecast == {"EURUSD": Market.FX}
