"""Tests of Q1 of 10-rq1-checks (``lib/_grid.py``): every model on the grid model's decisions."""

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.benchmark import Bootstrap, read_cache

KEYS = ["time", "asset", "horizon"]


def test_a_zero_shot_model_keeps_only_the_grid_models_decisions(
    checks: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    view = checks.grid_cache(cache, "sampled")
    skilled = view.forecasts("skilled")
    grid = cache.forecasts("sampled")[KEYS].drop_duplicates()
    assert len(skilled) == len(grid)
    assert skilled["time"].dt.hour.mod(4).eq(0).all()
    pd.testing.assert_frame_equal(
        skilled[KEYS].sort_values(KEYS).reset_index(drop=True),
        grid.sort_values(KEYS).reset_index(drop=True),
    )


def test_a_baseline_keeps_every_forecast(checks: ModuleType, quarters_cache_folder: Path):
    cache = read_cache(quarters_cache_folder)
    view = checks.grid_cache(cache, "sampled")
    assert len(view.forecasts("climatology")) == len(cache.forecasts("climatology"))


def test_every_model_decides_on_the_grid_models_grid(
    checks: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    view = checks.grid_cache(cache, "sampled")
    assert set(view.models["every_hours"]) == {4}
    assert set(cache.models["every_hours"]) == {0, 4}


def test_the_grid_models_own_forecasts_are_unchanged(
    checks: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    view = checks.grid_cache(cache, "sampled")
    pd.testing.assert_frame_equal(view.forecasts("sampled"), cache.forecasts("sampled"))


def test_a_model_that_decides_every_hour_is_no_grid(
    checks: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    with pytest.raises(ValueError, match="not a grid model"):
        checks.grid_cache(cache, "skilled")


def test_reopening_shares_count_the_first_bar_after_a_gap(checks: ModuleType, tmp_path: Path):
    # one asset with bars every hour but a 20-hour gap; the grid model decides at the reopening
    times = pd.date_range("2025-11-01", periods=10, freq="1h", tz="UTC")
    times = times.append(pd.date_range("2025-11-02 05:00", periods=10, freq="1h", tz="UTC"))
    labels = pd.DataFrame({"time": times, "asset": "AAPL", "horizon": 1, "window": "t1"})
    labels["label_time"] = labels["time"] + pd.Timedelta(hours=1)
    labels["label"], labels["log_return"], labels["sigma"] = 0.0, 0.0, 0.01
    folder = tmp_path / "cache"
    folder.mkdir()
    labels.to_parquet(folder / "labels.parquet")
    pd.DataFrame(
        {
            "model": ["grid"],
            "kind": ["zero-shot"],
            "release_date": ["2025-01-01"],
            "pretraining_cutoff": [""],
            "every_hours": [167],
        }
    ).to_csv(folder / "models.csv", index=False)
    pd.DataFrame({"asset": ["AAPL"], "market": ["stock"], "cost_per_side_bps": [1.0]}).to_csv(
        folder / "assets.csv", index=False
    )
    forecasts = labels.loc[[0, 10], KEYS].assign(seed=0)
    for column in [f"q{level}" for level in range(10, 100, 10)]:
        forecasts[column] = 0.0
    (folder / "forecasts").mkdir()
    forecasts.to_parquet(folder / "forecasts" / "grid.parquet")
    shares = checks.reopening_shares(
        read_cache(folder), "grid", windows=["t1"], min_closure_hours=12
    )
    row = shares.iloc[0]
    assert row["decisions"] == 20
    assert row["reopening_share"] == pytest.approx(1 / 20)
    assert row["grid_decisions"] == 2
    assert row["grid_reopening_share"] == pytest.approx(0.5)


def _one_asset_cache(folder: Path, gap_hours: int) -> Path:
    # one stock with ten hourly bars, a gap of `gap_hours` between close times, ten more bars;
    # a grid model "grid" decides at the first bar and at the first bar after the gap
    first = pd.date_range("2025-11-01", periods=10, freq="1h", tz="UTC")
    second = pd.date_range(first[-1] + pd.Timedelta(hours=gap_hours), periods=10, freq="1h")
    labels = pd.DataFrame({"time": first.append(second), "asset": "AAPL", "horizon": 1})
    labels["window"] = "t1"
    labels["label_time"] = labels["time"] + pd.Timedelta(hours=1)
    labels["label"], labels["log_return"], labels["sigma"] = 0.0, 0.0, 0.01
    folder.mkdir()
    labels.to_parquet(folder / "labels.parquet")
    pd.DataFrame(
        {
            "model": ["grid"],
            "kind": ["zero-shot"],
            "release_date": ["2025-01-01"],
            "pretraining_cutoff": [""],
            "every_hours": [167],
        }
    ).to_csv(folder / "models.csv", index=False)
    pd.DataFrame({"asset": ["AAPL"], "market": ["stock"], "cost_per_side_bps": [1.0]}).to_csv(
        folder / "assets.csv", index=False
    )
    forecasts = labels.loc[[0, 10], KEYS].assign(seed=0)
    for column in [f"q{level}" for level in range(10, 100, 10)]:
        forecasts[column] = 0.0
    (folder / "forecasts").mkdir()
    forecasts.to_parquet(folder / "forecasts" / "grid.parquet")
    return folder


@pytest.mark.parametrize(("gap_hours", "is_reopening"), [(12, False), (13, True), (20, True)])
def test_a_closure_needs_its_hours_without_a_bar(
    checks: ModuleType, tmp_path: Path, gap_hours: int, is_reopening: bool
):
    # close times 12 hours apart leave 11 hours without a bar: not a closure of 12 hours
    cache = read_cache(_one_asset_cache(tmp_path / "cache", gap_hours))
    row = checks.reopening_shares(cache, "grid", windows=["t1"], min_closure_hours=12).iloc[0]
    assert row["reopening_share"] == pytest.approx(1 / 20 if is_reopening else 0.0)
    assert row["grid_reopening_share"] == pytest.approx(0.5 if is_reopening else 0.0)
    # the decision at the last close before the gap (row 9) has its target across the closure;
    # the grid decides at rows 0 and 10, so none of its targets crosses it
    assert row["target_closure_share"] == pytest.approx(1 / 20 if is_reopening else 0.0)
    assert row["grid_target_closure_share"] == pytest.approx(0.0)


def test_the_gain_over_the_grid_model_is_positive_for_a_better_model(
    checks: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    view = checks.grid_cache(cache, "sampled")
    bootstrap = Bootstrap(level=0.95, n_resamples=200, seed=0, block_hours=24, min_block=4)
    gaps = checks.kronos_gaps(
        view,
        ["skilled", "climatology"],
        ["sampled"],
        windows=["t1"],
        horizon=1,
        bootstrap=bootstrap,
    )
    skilled = gaps[gaps["model"] == "skilled"]
    assert (skilled["gain"] > 0).all()
    assert (skilled["gain_low"] > 0).all()
    assert set(gaps["market"]) == {"crypto", "forex", "stock"}
