"""Tests of 13-context-baselines (``lib/_context.py``): baselines fitted on the last bars only."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.benchmark import QUANTILE_COLUMNS, read_cache
from src.models import QUANTILE_LEVELS


def test_the_random_walk_uses_only_the_previous_window(context: ModuleType):
    labels = np.array([1.0, -1.0, 3.0, 0.0, 10.0])
    quantiles = context.rolling_random_walk(labels, 2)
    assert np.isnan(quantiles[:2]).all()
    # decision 2 reads labels 0 and 1; decision 4 reads 2 and 3, never its own 10
    np.testing.assert_allclose(quantiles[2] / quantiles[2, -1], quantiles[4] / quantiles[4, -1])
    assert quantiles[2, -1] == pytest.approx(1.0 * 1.2815515655446004)
    assert quantiles[4, -1] == pytest.approx(np.sqrt(4.5) * 1.2815515655446004)


def test_climatology_takes_the_deciles_of_the_previous_window(context: ModuleType):
    labels = np.arange(12, dtype=float)
    quantiles = context.rolling_climatology(labels, 10)
    np.testing.assert_allclose(quantiles[10], np.quantile(labels[:10], QUANTILE_LEVELS))
    np.testing.assert_allclose(quantiles[11], np.quantile(labels[1:11], QUANTILE_LEVELS))


def test_the_ridge_learns_a_lag_inside_its_window(context: ModuleType):
    # the label is twice the last bar's vol-scaled return; sigma 1, so lag 1 is the return
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(400)
    labels = np.r_[np.nan, 2 * returns[:-1]]
    labels[0] = 0.0
    settings = context.ContextSettings(window=200, lags=(1,), ridge=1e-6, workers=1)
    quantiles = context.rolling_ridge(labels, returns, np.ones(400), np.zeros(400, bool), settings)
    median = quantiles[300:, 4]
    np.testing.assert_allclose(median, 2 * returns[299:399], atol=1e-3)


def test_garch_sees_no_return_of_its_own_decision(context: ModuleType):
    # a huge return after decision 300 must not change decision 300's spread, only later ones
    rng = np.random.default_rng(1)
    returns = 0.01 * rng.standard_normal(320)
    shocked = returns.copy()
    shocked[300] = 0.5
    days = np.repeat(np.arange(32), 10)
    settings = context.ContextSettings(window=200, lags=(1,), ridge=1.0, workers=1)
    with ProcessPoolExecutor(max_workers=1) as pool:
        calm, _ = context.garch_spreads(returns, days, settings, pool)
        hit, _ = context.garch_spreads(shocked, days, settings, pool)
    assert calm[300] == pytest.approx(hit[300])
    assert hit[301] > calm[301]
    assert np.isnan(calm[:200]).all()


def test_the_context_cache_serves_the_context_baselines_at_horizon_one(
    context: ModuleType, quarters_cache_folder: Path
):
    cache = read_cache(quarters_cache_folder)
    keys = cache.labels[cache.labels["horizon"] == 1][["time", "asset", "horizon"]].head(5)
    frame = keys.assign(seed=0)
    for column in QUANTILE_COLUMNS:
        frame[column] = 1.0
    view = context.context_cache(cache, {"climatology": frame}, ["t1", "t2"])
    assert set(view.labels["horizon"]) == {1}
    served = view.forecasts("climatology")
    assert len(served) == len(frame.merge(view.labels, on=["time", "asset", "horizon"]))
    np.testing.assert_allclose(served[list(QUANTILE_COLUMNS)].to_numpy(), 1.0)
    skilled = view.forecasts("skilled")
    assert set(skilled["horizon"]) == {1}


@pytest.mark.parametrize("name", ["climatology", "ridge"])
def test_a_later_label_never_changes_an_earlier_forecast(context: ModuleType, name: str):
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(300)
    labels = rng.standard_normal(300)
    settings = context.ContextSettings(window=100, lags=(1, 2), ridge=1.0, workers=1)

    def forecast(values: np.ndarray, moves: np.ndarray) -> np.ndarray:
        if name == "climatology":
            return context.rolling_climatology(values, 100)
        return context.rolling_ridge(values, moves, np.ones(300), np.zeros(300, bool), settings)

    before = forecast(labels, returns)
    changed_labels, changed_returns = labels.copy(), returns.copy()
    changed_labels[200], changed_returns[200] = 50.0, 50.0
    after = forecast(changed_labels, changed_returns)
    np.testing.assert_array_equal(before[:201], after[:201])
    assert not np.allclose(before[201:], after[201:], equal_nan=True)


def test_a_ridge_lag_across_a_gap_is_left_out(context: ModuleType):
    returns = np.ones(10)
    gaps = np.zeros(10, bool)
    gaps[5] = True
    features = context._context._lags(returns, np.ones(10), gaps, (1, 2))
    # decision 5 does not follow decision 4 by one bar, so its lag 1 (the bar before it) is
    # not the return of decision 4; decision 6's lag 2 reaches across the same gap
    assert np.isnan(features[5, 0])
    assert np.isnan(features[6, 1])
    assert features[6, 0] == pytest.approx(1.0)
    assert features[7, 1] == pytest.approx(2 / np.sqrt(2))
